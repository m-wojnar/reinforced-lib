#include <chrono>
#include <filesystem>
#include <map>
#include <string>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/ssid.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("RLibSim");

/***** Structures declarations *****/

struct FlowState
{
  uint64_t receivedBits = 0;
  uint64_t packetsNum = 0;
  uint64_t mcsSum = 0;
  double rateSum = 0.0;
  std::map<uint32_t, uint32_t> staFlows;
};

/***** Functions declarations *****/

void CwCallback (std::string path, u_int32_t oldValue, u_int32_t newValue);
void InstallTrafficGenerator (Ptr<ns3::Node> sourceNode, Ptr<ns3::Node> sinkNode, uint32_t port,
                              DataRate offeredLoad, uint32_t packetSize, double simulationTime);
void Measurement (Ptr<FlowMonitor> monitor, Ptr<Node> sinkNode, std::ostringstream *ostream);
void PhyRxOkCallback (Ptr<const Packet> packet, double snr, WifiMode mode, WifiPreamble preamble);
void PowerCallback (std::string path, Ptr<const Packet> packet, double txPowerW);
void StartMovement (Ptr<ns3::Node> node);
void WarmupMeasurement (Ptr<FlowMonitor> monitor);

/***** Global variables *****/

FlowState warmupState;
FlowState flowState;

uint64_t packetsNum;
uint64_t mcsSum;
double rateSum;

uint32_t channelWidth;
uint32_t minGI;
uint32_t nWifi;
uint32_t nss;
double velocity;
double warmupTime;
std::string wifiManagerName;

/***** Main with scenario definition *****/

int
main (int argc, char *argv[])
{
  // Initialize constants
  const double cooldownTime = 0.1;
  const uint32_t packetSize = 1500;

  // Initialize default simulation parameters
  channelWidth = 20;
  std::string csvPath = "";
  uint32_t dataRate = 125;
  double initialPosition = 0.0;
  double logEvery = 1.0;
  uint16_t memblockKey = 2333;
  minGI = 3200;
  nWifi = 1;
  nss = 1;
  std::string pcapPath = "";
  double simulationTime = 20.0;
  velocity = 0.0;
  warmupTime = 2.0;
  std::string wifiManager = "ns3::RLibWifiManager";
  wifiManagerName = "RLib";

  // Parse command line arguments
  CommandLine cmd;
  cmd.AddValue ("channelWidth", "Channel width (MHz)", channelWidth);
  cmd.AddValue ("csvPath", "Save an output file in the CSV format; relative path", csvPath);
  cmd.AddValue ("dataRate", "Aggregate traffic generators data rate (Mb/s)", dataRate);
  cmd.AddValue ("initialPosition", "Initial position of the AP on X axis (m)", initialPosition);
  cmd.AddValue ("logEvery", "Time interval between successive measurements (s)", logEvery);
  cmd.AddValue ("memblockKey", "ns3-ai shared memory id", memblockKey);
  cmd.AddValue ("minGI", "Shortest guard interval (ns)", minGI);
  cmd.AddValue ("nWifi", "Number of transmitting stations", nWifi);
  cmd.AddValue ("pcapPath", "Save a PCAP file from the AP; relative path", pcapPath);
  cmd.AddValue ("simulationTime", "Duration of the simulation; excluding warmup stage (s)",simulationTime);
  cmd.AddValue ("velocity", "Velocity of the AP on X axis (m/s)", velocity);
  cmd.AddValue ("warmupTime", "Duration of the warmup stage (s)", warmupTime);
  cmd.AddValue ("wifiManager", "Rate adaptation manager", wifiManager);
  cmd.AddValue ("wifiManagerName", "Name of the Wi-Fi manager in CSV", wifiManagerName);
  cmd.Parse (argc, argv);

  // Print simulation settings on the screen
  std::cout << std::endl
            << "Simulating an IEEE 802.11ax devices with the following settings:" << std::endl
            << "- frequency band: 5 GHz" << std::endl
            << "- max aggregated data rate: " << dataRate << " Mb/s" << std::endl
            << "- channel width: " << channelWidth << " Mhz" << std::endl
            << "- shortest guard interval: " << minGI << " ns" << std::endl
            << "- number of transmitting stations: " << nWifi << std::endl
            << "- packet size: " << packetSize << " B" << std::endl
            << "- rate adaptation manager: " << wifiManager << std::endl
            << "- simulation time: " << simulationTime << " s" << std::endl
            << "- warmup time: " << warmupTime << " s" << std::endl
            << "- initial AP position: " << initialPosition << " m" << std::endl
            << "- AP velocity: " << velocity << " m/s" << std::endl
            << std::endl;

  // Create AP and stations
  NodeContainer wifiApNode (1);
  NodeContainer wifiStaNodes (nWifi);

  // Configure wireless channel
  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  phy.SetChannel (channel.Create ());

  WifiMacHelper mac;
  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ax);

  // Setup rate adaptation algorithm
  if (wifiManager == "ns3::RLibWifiManager")
    {
      wifi.SetRemoteStationManager (wifiManager,
                                    "MemblockKey", UintegerValue (memblockKey),
                                    "nWifi", UintegerValue (nWifi));
    }
  else
    {
      wifi.SetRemoteStationManager (wifiManager);
    }

  // Create and configure Wi-Fi interfaces
  Ssid ssid = Ssid ("ns3-80211ax");
  mac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));

  NetDeviceContainer apDevice;
  apDevice = wifi.Install (phy, mac, wifiApNode);

  mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid));

  NetDeviceContainer staDevices;
  staDevices = wifi.Install (phy, mac, wifiStaNodes);

  // Set channel width and shortest GI
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelSettings",
               StringValue ("{0, " + std::to_string (channelWidth) + ", BAND_5GHZ, 0}"));

  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HeConfiguration/GuardInterval",
               TimeValue (NanoSeconds (minGI)));

  // Configure mobility and initial positions
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantVelocityMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNodes);

  wifiApNode.Get (0)
      ->GetObject<ConstantVelocityMobilityModel> ()
      ->SetPosition (Vector3D (initialPosition, 0.0, 0.0));

  Simulator::Schedule (Seconds (warmupTime), &StartMovement, wifiApNode.Get (0));

  // Install an Internet stack and configure IP addressing
  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiStaNodes);

  Ipv4AddressHelper address ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staNodesInterface = address.Assign (staDevices);
  Ipv4InterfaceContainer apNodeInterface = address.Assign (apDevice);

  // Configure applications
  DataRate applicationsDataRate (dataRate * 1e6 / nWifi);
  uint32_t portNumber = 9;

  for (uint32_t j = 0; j < nWifi; ++j)
    {
      InstallTrafficGenerator (wifiStaNodes.Get (j), wifiApNode.Get (0), portNumber++,
                               applicationsDataRate, packetSize, simulationTime);
    }

  //Install FlowMonitor
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll ();

  // Generate PCAP at AP
  if (!pcapPath.empty ())
    {
      phy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
      phy.EnablePcap (pcapPath, apDevice);
    }

  // Register callback for a successful packet reception
  uint32_t apNode = wifiApNode.Get (0)->GetId ();
  Config::ConnectWithoutContext ("/NodeList/" + std::to_string (apNode) +
                                     "/DeviceList/*/$ns3::WifiNetDevice/Phy/State/RxOk",
                                 MakeCallback (PhyRxOkCallback));

  // Register callbacks for the CW and power change
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/CwTrace",
                   MakeCallback (CwCallback));

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
                   MakeCallback (PowerCallback));

  // Schedule measurements
  Simulator::Schedule (Seconds (warmupTime), &WarmupMeasurement, monitor);

  std::ostringstream ostream;
  ostream << "wifiManager,seed,nWifi,channelWidth,minGI,velocity,position,time,meanMcs,meanRate,throughput"
          << std::endl;

  for (double time = warmupTime + logEvery; time <= warmupTime + simulationTime; time += logEvery)
    {
      Simulator::Schedule (Seconds (time), &Measurement, monitor, wifiApNode.Get (0), &ostream);
    }

  // Define simulation stop time
  Simulator::Stop (Seconds (warmupTime + simulationTime + cooldownTime));

  // Record start time
  std::cout << "Starting simulation..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now ();

  // Run the simulation!
  Simulator::Run ();

  // Record stop time and count duration
  auto finish = std::chrono::high_resolution_clock::now ();
  std::chrono::duration<double> elapsed = finish - start;

  std::cout << "Done!" << std::endl
            << "Elapsed time: " << elapsed.count () << " s" << std::endl
            << std::endl;

  // Calculate per-flow throughput, Jain's fairness index and print results
  double totalThr = 0;
  double jainsIndexN = 0.0;
  double jainsIndexD = 0.0;

  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
  std::cout << "Results: " << std::endl;

  for (const auto &stat : flowState.staFlows)
    {
      double stationThroughput = (stat.second - warmupState.staFlows[stat.first]) / (1e6 * simulationTime);
      totalThr += stationThroughput;

      jainsIndexN += stationThroughput;
      jainsIndexD += stationThroughput * stationThroughput;

      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (stat.first);
      std::cout << "Flow " << stat.first << " (" << t.sourceAddress << " -> "
                << t.destinationAddress << ")\tThroughput: " << stationThroughput << " Mb/s"
                << std::endl;
    }

  // Calculate mean rate, mean MCS and fairness index
  double allPackets = flowState.packetsNum - warmupState.packetsNum;
  double meanRate = (flowState.rateSum - warmupState.rateSum) / allPackets;
  double meanMcs = (flowState.mcsSum - warmupState.mcsSum) / allPackets;
  double fairnessIndex = jainsIndexN * jainsIndexN / (nWifi * jainsIndexD);

  // Print results
  std::cout << std::endl
            << "Network throughput: " << totalThr << " Mb/s" << std::endl
            << "Mean rate: " << meanRate << " Mb/s" << std::endl
            << "Mean MCS: " << meanMcs << std::endl
            << "Jain's fairness index: " << fairnessIndex << std::endl
            << std::endl;

  // Print results in CSV format
  if (!csvPath.empty ())
    {
      std::ofstream csvFile (csvPath);
      csvFile << ostream.str ();
      std::cout << ostream.str ();
    }

  //Clean-up
  Simulator::Destroy ();

  return 0;
}

/***** Function definitions *****/

void
CwCallback (std::string path, u_int32_t oldValue, u_int32_t newValue)
{
  size_t start = 10; // the length of the "/NodeList/" string
  size_t end = path.find ("/DeviceList/");
  std::string nodeId = path.substr (start, end - start);

  // Update CW value in the RLibWifiManager manager instance
  Config::Set ("/NodeList/" + nodeId + "/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
               "$ns3::RLibWifiManager/CW",
               UintegerValue (newValue));
}

void
InstallTrafficGenerator (Ptr<ns3::Node> sourceNode, Ptr<ns3::Node> sinkNode, uint32_t port,
                         DataRate offeredLoad, uint32_t packetSize, double simulationTime)
{
  // Get sink address
  Ptr<Ipv4> ipv4 = sinkNode->GetObject<Ipv4> ();
  Ipv4Address addr = ipv4->GetAddress (1, 0).GetLocal ();

  // Define type of service
  uint8_t tosValue = 0x70; //AC_BE

  // Add random fuzz to app start time
  Ptr<UniformRandomVariable> fuzz = CreateObject<UniformRandomVariable> ();
  fuzz->SetAttribute ("Min", DoubleValue (0.0));
  fuzz->SetAttribute ("Max", DoubleValue (warmupTime / 2));
  double applicationsStart = fuzz->GetValue ();

  // Configure source and sink
  InetSocketAddress sinkSocket (addr, port);
  sinkSocket.SetTos (tosValue);
  PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", sinkSocket);

  OnOffHelper onOffHelper ("ns3::UdpSocketFactory", sinkSocket);
  onOffHelper.SetConstantRate (offeredLoad, packetSize);

  // Configure applications
  ApplicationContainer sinkApplications (packetSinkHelper.Install (sinkNode));
  ApplicationContainer sourceApplications (onOffHelper.Install (sourceNode));

  sinkApplications.Start (Seconds (applicationsStart));
  sinkApplications.Stop (Seconds (warmupTime + simulationTime));
  sourceApplications.Start (Seconds (applicationsStart));
  sourceApplications.Stop (Seconds (warmupTime + simulationTime));
}

void
Measurement (Ptr<FlowMonitor> monitor, Ptr<Node> sinkNode, std::ostringstream *ostream)
{
  // Initial metrics values
  static double lastTime = warmupTime;
  static uint64_t lastReceivedBits = warmupState.receivedBits;
  static uint64_t lastPacketsNum = warmupState.packetsNum;
  static uint64_t lastMcsSum = warmupState.mcsSum;
  static double lastRateSum = warmupState.rateSum;

  // Calculate and save metrics since the previous measurement
  double currentTime = Simulator::Now ().GetSeconds ();
  uint64_t receivedBits = 0;

  for (auto &stat : monitor->GetFlowStats ())
    {
      uint64_t bits = 8 * stat.second.rxBytes;
      receivedBits += bits;
      flowState.staFlows[stat.first] = bits;
    }

  flowState.receivedBits = receivedBits;
  flowState.packetsNum = packetsNum;
  flowState.mcsSum = mcsSum;
  flowState.rateSum = rateSum;

  double throughput = (receivedBits - lastReceivedBits) / (1e6 * (currentTime - lastTime));
  double meanMcs = (mcsSum - lastMcsSum) / (double) (packetsNum - lastPacketsNum);
  double meanRate = (rateSum - lastRateSum) / (packetsNum - lastPacketsNum);

  if (packetsNum == lastPacketsNum)
    {
      meanMcs = meanRate = 0;
    }

  lastTime = currentTime;
  lastReceivedBits = receivedBits;
  lastPacketsNum = packetsNum;
  lastMcsSum = mcsSum;
  lastRateSum = rateSum;

  // Get current AP position
  double position = sinkNode->GetObject<MobilityModel> ()->GetPosition ().x;

  // Add current state to CSV
  (*ostream) << wifiManagerName << ',' << RngSeedManager::GetRun () << ',' << nWifi << ','
             << channelWidth << ',' << minGI << ',' << velocity << ',' << position << ','
             << currentTime - warmupTime << ',' << meanMcs << ',' << meanRate << ',' << throughput
             << std::endl;
}

void
PhyRxOkCallback (Ptr<const Packet> packet, double snr, WifiMode mode, WifiPreamble preamble)
{
  if (preamble == WifiPreamble::WIFI_PREAMBLE_HE_SU)
    {
      packetsNum++;
      rateSum += mode.GetDataRate (channelWidth, minGI, nss) / 1e6;
      mcsSum += mode.GetMcsValue ();
    }
}

void
PowerCallback (std::string path, Ptr<const Packet> packet, double txPowerW)
{
  size_t start = 10; // the length of the "/NodeList/" string
  size_t end = path.find ("/DeviceList/");
  std::string nodeId = path.substr (start, end - start);

  // Update power level in the RLibWifiManager manager instance
  Config::Set ("/NodeList/" + nodeId + "/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
               "$ns3::RLibWifiManager/Power",
               DoubleValue (10 * (3 + log10 (txPowerW))));
}

void
StartMovement (Ptr<Node> node)
{
  node->GetObject<ConstantVelocityMobilityModel> ()->SetVelocity (Vector3D (velocity, 0.0, 0.0));
}

void
WarmupMeasurement (Ptr<FlowMonitor> monitor)
{
  for (auto &stat : monitor->GetFlowStats ())
    {
      uint32_t bits = 8 * stat.second.rxBytes;
      warmupState.staFlows.insert (std::pair<uint32_t, uint32_t> (stat.first, bits));
      warmupState.receivedBits += bits;
    }

  warmupState.packetsNum = packetsNum;
  warmupState.mcsSum = mcsSum;
  warmupState.rateSum = rateSum;
}
