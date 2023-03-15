#ifndef SCENARIO_H
#define SCENARIO_H

#include <fstream>
#include <string>
#include <math.h>
#include <ctime>   //timestampi
#include <iomanip> // put_time
#include <deque>
#include <algorithm>

using namespace std;
using namespace ns3;

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
#endif