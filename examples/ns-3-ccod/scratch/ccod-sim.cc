/*
MIT License

Copyright (c) 2023 Witold Wydmański

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



The program below is a modified version of the original authored by Witold Wydmański,
which can be found at https://github.com/wwydmanski/RLinWiFi/blob/master/linear-mesh/cw.cc.

The original code was for the following research article:
W. Wydmański and S. Szott, "Contention Window Optimization in IEEE 802.11ax Networks 
with Deep Reinforcement Learning" (URL: https://ieeexplore.ieee.org/document/9417575),
2021 IEEE Wireless Communications and Networking Conference (WCNC), 2021,
doi: 10.1109/WCNC49053.2021.9417575.
*/

#include <chrono>
#include <filesystem>
#include <map>
#include <regex>
#include <string>
#include <fstream>
#include <math.h>
#include <ctime>   //timestampi
#include <iomanip> // put_time
#include <deque>
#include <algorithm>
#include <csignal>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/node-list.h"
#include "ns3/propagation-module.h"
#include "ns3/ssid.h"
#include "ns3/wifi-module.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ns3-ai-module.h"

#define DEFAULT_MEMBLOCK_KEY 2333
#define MAX_HISTORY_LENGTH 512

using namespace std;
using namespace ns3;

bool dry_run = false;

/***** ns3-ai structures *****/

// ns3-ai structures

struct Env
{
    float history[MAX_HISTORY_LENGTH];
    float reward;
} Packed;

struct Act
{
    float action;
} Packed;

Ns3AIRL<Env, Act> * m_env = new Ns3AIRL<Env, Act> (DEFAULT_MEMBLOCK_KEY);

/***** scenario.h *****/

class Scenario
{
  protected:
    int nWifim;
    NodeContainer wifiStaNode;
    NodeContainer wifiApNode;
    int port;
    std::string offeredLoad;
    std::vector<double> start_times;
    std::vector<double> end_times;
    int history_length;

    void installTrafficGenerator(ns3::Ptr<ns3::Node> fromNode,
                                 ns3::Ptr<ns3::Node> toNode,
                                 int port,
                                 std::string offeredLoad,
                                 double startTime,
                                 double endTime,
                                 ns3::Callback<void, Ptr<const Packet>> callback);

  public:
    Scenario(int nWifim, NodeContainer wifiStaNode, NodeContainer wifiApNode, int port, std::string offeredLoad, int history_length);
    virtual void installScenario(double simulationTime, double envStepTime, ns3::Callback<void, Ptr<const Packet>> callback) = 0;
    void PopulateARPcache();
    int getActiveStationCount(double time);
    float getStationUptime(int id, double time);
};

class BasicScenario : public Scenario
{
    using Scenario::Scenario;

  public:
    void installScenario(double simulationTime, double envStepTime, ns3::Callback<void, Ptr<const Packet>> callback) override;
};

class ConvergenceScenario : public Scenario
{
    using Scenario::Scenario;

  public:
    void installScenario(double simulationTime, double envStepTime, ns3::Callback<void, Ptr<const Packet>> callback) override;
};

class ScenarioFactory
{
  private:
    int nWifim;
    NodeContainer wifiStaNode;
    NodeContainer wifiApNode;
    int port;
    int history_length;
    std::string offeredLoad;

  public:
    ScenarioFactory(int nWifim, NodeContainer wifiStaNode, NodeContainer wifiApNode, int port, std::string offeredLoad, int history_length)
    {
        this->nWifim = nWifim;
        this->wifiStaNode = wifiStaNode;
        this->wifiApNode = wifiApNode;
        this->port = port;
        this->offeredLoad = offeredLoad;
        this->history_length = history_length;
    }

    Scenario *getScenario(std::string scenario)
    {
        Scenario *wifiScenario;
        if (scenario == "basic")
        {
            wifiScenario = new BasicScenario(this->nWifim, this->wifiStaNode, this->wifiApNode, this->port, this->offeredLoad, this->history_length);
        }
        else if (scenario == "convergence")
        {
            wifiScenario = new ConvergenceScenario(this->nWifim, this->wifiStaNode, this->wifiApNode, this->port, this->offeredLoad, this->history_length);
        }
        else
        {
            std::cout << "Unsupported scenario" << endl;
            exit(0);
        }
        return wifiScenario;
    }
};

Scenario::Scenario(int nWifim, NodeContainer wifiStaNode, NodeContainer wifiApNode, int port, std::string offeredLoad, int history_length)
{
    this->nWifim = nWifim;
    this->wifiStaNode = wifiStaNode;
    this->wifiApNode = wifiApNode;
    this->port = port;
    this->offeredLoad = offeredLoad;
    this->history_length = history_length;
}

int Scenario::getActiveStationCount(double time)
{
    int res=0;
    for(uint i=0; i<start_times.size(); i++)
        if(start_times.at(i)<time && time<end_times.at(i))
            res++;
    return res;
}

float Scenario::getStationUptime(int id, double time)
{
    return time - start_times.at(id);
    // int res=0;
    // for(uint i=0; i<start_times.size(); i++)
    //     if(start_times.at(i)<time && time<end_times.at(i))
    //         res++;
    // return res;
}

void Scenario::installTrafficGenerator(Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, int port, string offeredLoad, double startTime, double endTime,
                                       ns3::Callback<void, Ptr<const Packet>> callback)
{
    start_times.push_back(startTime);
    end_times.push_back(endTime);

    Ptr<Ipv4> ipv4 = toNode->GetObject<Ipv4>();           // Get Ipv4 instance of the node
    Ipv4Address addr = ipv4->GetAddress(1, 0).GetLocal(); // Get Ipv4InterfaceAddress of xth interface.

    ApplicationContainer sourceApplications, sinkApplications;

    uint8_t tosValue = 0x70; //AC_BE
    //Add random fuzz to app start time
    double min = 0.0;
    double max = 1.0;
    Ptr<UniformRandomVariable> fuzz = CreateObject<UniformRandomVariable>();
    fuzz->SetAttribute("Min", DoubleValue(min));
    fuzz->SetAttribute("Max", DoubleValue(max));

    InetSocketAddress sinkSocket(addr, port);
    sinkSocket.SetTos(tosValue);
    //OnOffHelper onOffHelper ("ns3::TcpSocketFactory", sinkSocket);
    OnOffHelper onOffHelper("ns3::UdpSocketFactory", sinkSocket);
    onOffHelper.SetConstantRate(DataRate(offeredLoad + "Mbps"), 1500 - 20 - 8 - 8);
    // onOffHelper.TraceConnectWithoutContext("Tx", MakeCallback(&packetSent));
    sourceApplications.Add(onOffHelper.Install(fromNode)); //fromNode

    //PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", sinkSocket);
    // PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", sinkSocket);
    UdpServerHelper sink(port);
    sinkApplications = sink.Install(toNode);
    // sinkApplications.Add (packetSinkHelper.Install (toNode)); //toNode

    sinkApplications.Start(Seconds(startTime));
    sinkApplications.Stop(Seconds(endTime));

    Ptr<UdpServer> udpServer = DynamicCast<UdpServer>(sinkApplications.Get(0));
    udpServer->TraceConnectWithoutContext("Rx", callback);

    sourceApplications.Start(Seconds(startTime));
    sourceApplications.Stop(Seconds(endTime));
}

void Scenario::PopulateARPcache()
{
    Ptr<ArpCache> arp = CreateObject<ArpCache>();
    arp->SetAliveTimeout(Seconds(3600 * 24 * 365));

    for (NodeList::Iterator i = NodeList::Begin(); i != NodeList::End(); ++i)
    {
        Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol>();
        NS_ASSERT(ip != 0);
        ObjectVectorValue interfaces;
        ip->GetAttribute("InterfaceList", interfaces);

        for (ObjectVectorValue::Iterator j = interfaces.Begin(); j != interfaces.End(); j++)
        {
            Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface>();
            NS_ASSERT(ipIface != 0);
            Ptr<NetDevice> device = ipIface->GetDevice();
            NS_ASSERT(device != 0);
            Mac48Address addr = Mac48Address::ConvertFrom(device->GetAddress());

            for (uint32_t k = 0; k < ipIface->GetNAddresses(); k++)
            {
                Ipv4Address ipAddr = ipIface->GetAddress(k).GetLocal();
                if (ipAddr == Ipv4Address::GetLoopback())
                    continue;

                ArpCache::Entry *entry = arp->Add(ipAddr);
                Ipv4Header ipv4Hdr;
                ipv4Hdr.SetDestination(ipAddr);
                Ptr<Packet> p = Create<Packet>(100);
                entry->MarkWaitReply(ArpCache::Ipv4PayloadHeaderPair(p, ipv4Hdr));
                entry->MarkAlive(addr);
            }
        }
    }

    for (NodeList::Iterator i = NodeList::Begin(); i != NodeList::End(); ++i)
    {
        Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol>();
        NS_ASSERT(ip != 0);
        ObjectVectorValue interfaces;
        ip->GetAttribute("InterfaceList", interfaces);

        for (ObjectVectorValue::Iterator j = interfaces.Begin(); j != interfaces.End(); j++)
        {
            Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface>();
            ipIface->SetAttribute("ArpCache", PointerValue(arp));
        }
    }
}

void BasicScenario::installScenario(double simulationTime, double envStepTime, ns3::Callback<void, Ptr<const Packet>> callback)
{
    for (int i = 0; i < this->nWifim; ++i)
    {
        installTrafficGenerator(this->wifiStaNode.Get(i), this->wifiApNode.Get(0), this->port++, this->offeredLoad, 0.0, simulationTime + 2 + envStepTime*history_length, callback);
    }
}

void ConvergenceScenario::installScenario(double simulationTime, double envStepTime, ns3::Callback<void, Ptr<const Packet>> callback)
{
    float delta = simulationTime/(this->nWifim-4);
    float delay = history_length*envStepTime;
    if (this->nWifim > 5)
    {
        for (int i = 0; i < 5; ++i)
        {
            installTrafficGenerator(this->wifiStaNode.Get(i), this->wifiApNode.Get(0), this->port++, this->offeredLoad, 0.0 , simulationTime + 2 + delay, callback);
        }
        for (int i = 5; i < this->nWifim; ++i)
        {
            installTrafficGenerator(this->wifiStaNode.Get(i), this->wifiApNode.Get(0), this->port++, this->offeredLoad, delay+(i - 4) * delta, simulationTime + 2 + delay, callback);
        }
    }
    else
    {
        std::cout << "Not enough Wi-Fi stations to support the convergence scenario." << endl;
        exit(0);
    }
}

/***** Functions declarations *****/

bool execute_action(float action);
float getCurrentThr(void);
float getReward(void);
double jain_index(void);
void packetReceived(Ptr<const Packet> packet);
void packetSent(Ptr<const Packet> packet, double txPowerW);
void recordHistory();
void ScheduleNextStateRead(double envStepTime);
void set_nodes(int channelWidth, int guardInterval, int rng, int32_t simSeed, NodeContainer wifiStaNode, NodeContainer wifiApNode, YansWifiPhyHelper phy, WifiMacHelper mac, WifiHelper wifi, NetDeviceContainer &apDevice);
void set_phy(int nWifi, NodeContainer &wifiStaNode, NodeContainer &wifiApNode, YansWifiPhyHelper &phy);
void set_sim(bool tracing, bool dry_run, int warmup, YansWifiPhyHelper phy, NetDeviceContainer apDevice, int end_delay, Ptr<FlowMonitor> &monitor, FlowMonitorHelper &flowmon);
void signalHandler(int signum);

/***** Global variables *****/

double envStepTime = 0.1;
double simulationTime = 10; //seconds
// double current_time = 0.0;
bool verbose = false;
int end_delay = 0;

Ptr<FlowMonitor> monitor;
FlowMonitorHelper flowmon;
ofstream outfile ("CW_data.csv", fstream::out);
string rewardCsvPath = "";
std::ostringstream rewardCsvOutput;

uint32_t CW = 0;

uint32_t history_length = 20;
deque<float> history;

string type = "discrete";
bool non_zero_start = false;
Scenario *wifiScenario;

uint64_t g_rxPktNum = 0;
uint64_t g_txPktNum = 0;

/***** Main with scenario definition *****/

int
main (int argc, char *argv[])
{
    int nWifi = 5;
    bool tracing = false;
    bool useRts = false;
    int mcs = 11;
    int channelWidth = 20;
    int guardInterval = 800;
    string offeredLoad = "150";
    int port = 1025;
    string outputCsv = "cw.csv";
    string scenario = "basic";
    dry_run = false;

    int rng = 42;
    int warmup = 1;

    int32_t simSeed = -1;

    signal(SIGTERM, signalHandler);
    outfile << "SimulationTime,CW" << endl;

    CommandLine cmd;
    cmd.AddValue("agentType", "Type of agent actions: discrete, continuous", type);
    cmd.AddValue("csvPath", "Save an reward output file in the CSV format", rewardCsvPath);
    cmd.AddValue("CW", "Value of Contention Window", CW);
    cmd.AddValue("dryRun", "Execute scenario with BEB and no agent interaction", dry_run);
    cmd.AddValue("envStepTime", "Step time in seconds. Default: 0.1s", envStepTime);
    cmd.AddValue("historyLength", "Length of history window", history_length);
    cmd.AddValue("nonZeroStart", "Start only after history buffer is filled", non_zero_start);
    cmd.AddValue("nWifi", "Number of wifi 802.11ax STA devices", nWifi);
    // cmd.AddValue("rng", "Number of RngRun", rng);
    cmd.AddValue("scenario", "Scenario for analysis: basic, convergence, reaction", scenario);
    // cmd.AddValue("seed", "Random seed", simSeed);
    cmd.AddValue("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
    cmd.AddValue("tracing", "Enable pcap tracing", tracing);
    cmd.AddValue("verbose", "Tell echo applications to log if true", verbose);

    cmd.Parse(argc, argv);

    if (history_length > MAX_HISTORY_LENGTH) {
        cout << "Error, Maximum history length " << MAX_HISTORY_LENGTH << " exceeded!" << endl;
        exit(1);
    }

    cout << endl
         << "Ns3Env parameters:" << endl
         << "--agentType: " << type << endl
         << "--dryRun: " << dry_run << endl
         << "--envStepTime: " << envStepTime << endl
         << "--historyLength: " << history_length << endl
         << "--nonZeroStart: " << non_zero_start << endl
         << "--nWifi: " << nWifi << endl
         << "--scenario: " << scenario << endl
         << "--RngRun: " << RngSeedManager::GetRun () << endl
         << "--simulationTime: " << simulationTime << endl << endl;
    
    // Set the ns3-ai operation lock
    if (!dry_run) {
        m_env->SetCond(2, 0);
    }

    if (verbose)
    {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }

    if (useRts)
    {
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue("0"));
    }

    NodeContainer wifiStaNode;
    NodeContainer wifiApNode;
    YansWifiPhyHelper phy;
    set_phy(nWifi, wifiStaNode, wifiApNode, phy);

    WifiMacHelper mac;
    WifiHelper wifi;

    wifi.SetStandard (WIFI_STANDARD_80211ax);

    std::ostringstream oss;
    oss << "HeMcs" << mcs;
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue(oss.str()),
                                 "ControlMode", StringValue(oss.str()));

    //802.11ac PHY
    /*
  phy.Set ("ShortGuardEnabled", BooleanValue (0));
  wifi.SetStandard (WIFI_PHY_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
  "DataMode", StringValue ("VhtMcs8"),
  "ControlMode", StringValue ("VhtMcs8"));
 */
    //802.11n PHY
    //phy.Set ("ShortGuardEnabled", BooleanValue (1));
    //wifi.SetStandard (WIFI_PHY_STANDARD_80211n_5GHZ);
    //wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
    //                              "DataMode", StringValue ("HtMcs7"),
    //                              "ControlMode", StringValue ("HtMcs7"));

    NetDeviceContainer apDevice;
    set_nodes(channelWidth, guardInterval, rng, simSeed, wifiStaNode, wifiApNode, phy, mac, wifi, apDevice);

    ScenarioFactory helper = ScenarioFactory(nWifi, wifiStaNode, wifiApNode, port, offeredLoad, history_length);
    wifiScenario = helper.getScenario(scenario);

    // if (!dry_run)
    // {
    if (non_zero_start)
        end_delay = envStepTime * history_length + 1.0;
    else
        end_delay = 0.0;
    // }

    wifiScenario->installScenario(simulationTime + end_delay + envStepTime, envStepTime, MakeCallback(&packetReceived));

    // Config::ConnectWithoutContext("/NodeList/0/ApplicationList/*/$ns3::OnOffApplication/Tx", MakeCallback(&packetSent));
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin", MakeCallback(packetSent));

    wifiScenario->PopulateARPcache();
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    set_sim(tracing, dry_run, warmup, phy, apDevice, end_delay, monitor, flowmon);

    double flowThr;
    float res =  g_rxPktNum * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024;
    cout << endl << "Sent mbytes: " << res << "\tThroughput: " << res/simulationTime << endl;
    ofstream myfile;
    myfile.open(outputCsv, ios::app);

    /* Contents of CSV output file
    Timestamp, CW, nWifi, RngRun, SourceIP, DestinationIP, Throughput
    */
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        auto time = std::time(nullptr); //Get timestamp
        auto tm = *std::localtime(&time);
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        flowThr = i->second.rxBytes * 8.0 / simulationTime / 1000 / 1000;
        cout << "Flow " << i->first << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\tThroughput: " << flowThr << " Mbps\tTime: " << i->second.timeLastRxPacket.GetSeconds() - i->second.timeFirstTxPacket.GetSeconds() << " s\tRx packets " << i->second.rxPackets << endl;
        myfile << std::put_time(&tm, "%Y-%m-%d %H:%M") << "," << CW << "," << nWifi << "," << RngSeedManager::GetRun() << "," << t.sourceAddress << "," << t.destinationAddress << "," << flowThr;
        myfile << std::endl;
    }
    myfile.close();

    // Save reward file in CSV format
    if (dry_run)
    {
        std::string rewardCsvString = rewardCsvOutput.str ();
        if (!rewardCsvPath.empty ())
        {
            std::ofstream csvFile (rewardCsvPath);
            csvFile << rewardCsvString;
        }
    }

    Simulator::Destroy();
    cout << "Packets registered by handler: " << g_rxPktNum << " Packets" << endl;

    return 0;
}

/***** Function definitions *****/

bool
execute_action(float action)
{
    if (verbose)
        NS_LOG_UNCOND("Action executed: " << action);
    
    if (type == "discrete")
    {
        CW = pow(2, 4+action);
    }
    else if (type == "continuous")
    {
        CW = pow(2, action + 4);
    }
    else if (type == "direct_continuous")
    {
        CW = action;
    }
    else
    {
        std::cout << "Unsupported agent type!" << endl;
        exit(1);
    }

    uint32_t min_cw = 16;
    uint32_t max_cw = 1024;

    CW = min(max_cw, max(CW, min_cw));
    outfile << Simulator::Now().GetSeconds() << "," << CW << endl;

    if(!dry_run){
        Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MinCw", UintegerValue(CW));
        Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MaxCw", UintegerValue(CW));
    }
    return true;
}

float
getReward(void)
{
    static float ticks = 0.0;
    static uint32_t last_packets = 0;
    static float last_reward = 0.0;
    ticks += envStepTime;

    float res = g_rxPktNum - last_packets;
    float reward = res * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024 / (5 * 150 * envStepTime) * 10;

    last_packets = g_rxPktNum;

    if (ticks <= 2 * envStepTime)
        return 0.0;

    if (verbose)
        NS_LOG_UNCOND("Reward: " << reward);

    if(reward>1.0f || reward<0.0f)
        reward = last_reward;
    last_reward = reward;
    return last_reward;
}

double
jain_index(void)
{
    double flowThr;
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    double nominator = 0;
    double denominator = 0;
    double n = 0;
    double station_id = 0;
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        flowThr = i->second.rxBytes;
        flowThr /= wifiScenario->getStationUptime(station_id, Simulator::Now ().GetSeconds ());
        if(flowThr>0){
            nominator += flowThr;
            denominator += flowThr*flowThr;
            n++;
        }
        station_id++;
    }
    nominator *= nominator;
    denominator *= n;
    return nominator/denominator;
}

void
packetReceived(Ptr<const Packet> packet)
{
    NS_LOG_DEBUG("Client received a packet of " << packet->GetSize() << " bytes");
    g_rxPktNum++;
}

void
packetSent(Ptr<const Packet> packet, double txPowerW)
{
    g_txPktNum++;
}

void
recordHistory()
{
    // Keep track of the observations
    // We will define them as the error rate of the last `envStepTime` seconds
    static uint32_t last_rx = 0;            // Previously received packets
    static uint32_t last_tx = 0;            // Previously transmitted packets
    static uint32_t calls = 0;              // Number of calls to this function
    calls++;
    // current_time += envStepTime;

    float received = g_rxPktNum - last_rx;  // Received packets since the last observation
    float sent = g_txPktNum - last_tx;      // Sent (...)
    float errs = sent - received;           // Errors (...)
    float ratio;

    if (sent == 0)
        ratio = 0.0;
    else
        ratio = errs / sent;

    history.push_front(ratio);

    // Remove the oldest observation if we have filled the history
    if (history.size() > history_length)
    {
        history.pop_back();
    }

    // Replace the last observation with the current one
    last_rx = g_rxPktNum;
    last_tx = g_txPktNum;

    if (calls < history_length && non_zero_start)
    {   
        // Schedule the next observation if we are not at the end of the simulation
        Simulator::Schedule(Seconds(envStepTime), &recordHistory);
    }
    else if (calls == history_length && non_zero_start)
    {
        g_rxPktNum = 0;
        g_txPktNum = 0;
        last_rx = 0;
        last_tx = 0;
    }
}

void
ScheduleNextStateRead(double envStepTime)
{
    Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime);

    static double curr_time = 0.0;

    recordHistory();
    float reward = getReward();

    if (!dry_run)
    {
        // Here is the ns3-ai communication with python agent
        // 1. push history and reward to DQN agent as observation
        auto env = m_env->EnvSetterCond();
        for (int i = 0; i < history.size(); i++) {
            env->history[i] = history.at(i);
        }
        env->reward = reward;
        m_env->SetCompleted();

            // 2. get action from DQN agent
        auto act = m_env->ActionGetterCond();
        float current_action = act->action;
        m_env->GetCompleted();

        // Act according to the retrived action
        execute_action((float) current_action);
    }
    else
    {
        rewardCsvOutput << "CSMA," << curr_time << "," << reward << endl;
    }

    curr_time += envStepTime;
}

void
set_nodes(int channelWidth, int guardInterval, int rng, int32_t simSeed, NodeContainer wifiStaNode, NodeContainer wifiApNode, YansWifiPhyHelper phy, WifiMacHelper mac, WifiHelper wifi, NetDeviceContainer &apDevice)
{
    // Set the access point details
    Ssid ssid = Ssid("ns3-80211ax");

    mac.SetType("ns3::StaWifiMac",
                "Ssid", SsidValue(ssid),
                "ActiveProbing", BooleanValue(false),
                "BE_MaxAmpduSize", UintegerValue(0),
                "MaxMissedBeacons", UintegerValue (1000));  // prevents exhaustion of association IDs

    NetDeviceContainer staDevice;
    staDevice = wifi.Install(phy, mac, wifiStaNode);

    mac.SetType("ns3::ApWifiMac",
                "EnableBeaconJitter", BooleanValue(false),
                "Ssid", SsidValue(ssid));

    apDevice = wifi.Install(phy, mac, wifiApNode);

    // Set channel width and shortest GI
    Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HeConfiguration/GuardInterval",
                TimeValue (NanoSeconds (guardInterval)));
    Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelSettings",
                StringValue ("{0, " + std::to_string (channelWidth) + ", BAND_5GHZ, 0}"));

    // mobility.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(1.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNode);

    /* Internet stack*/
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNode);

    //Random
    //if(simSeed!=-1)
    //    RngSeedManager::SetSeed(simSeed);
    //RngSeedManager::SetRun(rng);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staNodeInterface;
    Ipv4InterfaceContainer apNodeInterface;

    staNodeInterface = address.Assign(staDevice);
    apNodeInterface = address.Assign(apDevice);

    if (!dry_run)
    {
        Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MinCw", UintegerValue(CW));
        Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MaxCw", UintegerValue(CW));
    }
    else
    {
        NS_LOG_UNCOND("Default CW");
        Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MinCw", UintegerValue(16));
        Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MaxCw", UintegerValue(1024));
    }
}

void
set_phy(int nWifi, NodeContainer &wifiStaNode, NodeContainer &wifiApNode, YansWifiPhyHelper &phy)
{
    Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel>();
    lossModel->SetDefaultLoss(50);

    wifiStaNode.Create(nWifi);
    wifiApNode.Create(1);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
    Ptr<YansWifiChannel> chan = channel.Create();
    chan->SetPropagationLossModel(lossModel);
    chan->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());
    phy.SetChannel(chan);
}

void
set_sim(bool tracing, bool dry_run, int warmup, YansWifiPhyHelper phy, NetDeviceContainer apDevice, int end_delay, Ptr<FlowMonitor> &monitor, FlowMonitorHelper &flowmon)
{
    monitor = flowmon.InstallAll();
    monitor->SetAttribute("StartTime", TimeValue(Seconds(warmup)));

    if (tracing)
    {
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        phy.EnablePcap("cw", apDevice.Get(0));
    }

    // if (!dry_run)
    // {
    if (non_zero_start)
    {

        Simulator::Schedule(Seconds(1.0), &recordHistory);
        Simulator::Schedule(Seconds(envStepTime * history_length + 1.0), &ScheduleNextStateRead, envStepTime);
    }
    else
        Simulator::Schedule(Seconds(1.0), &ScheduleNextStateRead, envStepTime);
    // }

    Simulator::Stop(Seconds(simulationTime + end_delay + 1.0 + envStepTime*(history_length+1)));

    cout << "Simulation started..." << endl;
    Simulator::Run();
}

void signalHandler(int signum)
{
    cout << "Interrupt signal " << signum << " received.\n";
    exit(signum);
}
