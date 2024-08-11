#include <chrono>
#include <filesystem>
#include <map>
#include <regex>
#include <string>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/node-list.h"
#include "ns3/ssid.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/rlib-wifi-manager.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("RLibRA");

/***** Structures declarations *****/

struct FlowState
{
  uint64_t receivedBits = 0;
  std::map<uint32_t, uint32_t> staFlows;
};

/***** Functions declarations *****/

void ChangePower (NodeContainer wifiStaNodes, uint8_t powerLevel);
void CwCallback (std::string path, u_int32_t value, u_int8_t linkId);
void InstallTrafficGenerator (Ptr<ns3::Node> sourceNode, Ptr<ns3::Node> sinkNode, uint32_t port,
                              DataRate offeredLoad, uint32_t packetSize, double simulationTime,
                              double warmupTime);
void Measurement (Ptr<FlowMonitor> monitor, Ptr<Node> sinkNode, double warmupTime, double nextMeasurement);
void PowerCallback (std::string path, Ptr<const Packet> packet, double txPowerW);
void StartMovement (Ptr<ns3::Node> node, double velocity);
void WarmupMeasurement (Ptr<FlowMonitor> monitor);

/***** Global variables *****/

FlowState warmupState;
FlowState flowState;

std::ostringstream csvOutput;
std::string csvPrefix;

RLibWifiManager rlib;

/***** Main with scenario definition *****/

int
main (int argc, char *argv[])
{
  // Initialize constants
  const double defaultPower = 16.0206;
  const double cooldownTime = 0.1;
  const uint32_t packetSize = 1500;
  const uint32_t nss = 1;

  // Initialize default simulation parameters
  std::string csvPath = "";
  std::string pcapPath = "";
  std::string wifiManager = "ns3::RLibWifiManager";
  std::string wifiManagerName = "";

  std::string lossModel = "LogDistance";
  uint32_t channelWidth = 20;
  uint32_t dataRate = 125;
  uint32_t minGI = 3200;
  uint32_t nWifi = 1;
  double deltaPower = 0.;
  double intervalPower = 4.;

  std::string mobilityModel = "Distance";
  double area = 40.;
  double nodeSpeed = 1.4;
  double nodePause = 20.;
  double initialPosition = 0.;
  double velocity = 0.;

  double simulationTime = 20.;
  double warmupTime = 2.;
  double logEvery = 1.;

  // Parse command line arguments
  CommandLine cmd;
  cmd.AddValue ("area", "Size of the square in which stations are wandering (m) [RWPM mobility type]", area);
  cmd.AddValue ("channelWidth", "Channel width (MHz)", channelWidth);
  cmd.AddValue ("csvPath", "Save an output file in the CSV format", csvPath);
  cmd.AddValue ("dataRate", "Aggregate traffic generators data rate (Mb/s)", dataRate);
  cmd.AddValue ("deltaPower", "Power change (dBm)", deltaPower);
  cmd.AddValue ("initialPosition", "Initial position of the AP on X axis (m) [Distance mobility type]", initialPosition);
  cmd.AddValue ("intervalPower", "Interval between power change (s)", intervalPower);
  cmd.AddValue ("logEvery", "Time interval between successive measurements (s)", logEvery);
  cmd.AddValue ("lossModel", "Propagation loss model to use [LogDistance, Nakagami]", lossModel);
  cmd.AddValue ("minGI", "Shortest guard interval (ns)", minGI);
  cmd.AddValue ("mobilityModel", "Mobility model [Distance, RWPM]", mobilityModel);
  cmd.AddValue ("nodeSpeed", "Maximum station speed (m/s) [RWPM mobility type]", nodeSpeed);
  cmd.AddValue ("nodePause", "Maximum time station waits in newly selected position (s) [RWPM mobility type]", nodePause);
  cmd.AddValue ("nWifi", "Number of transmitting stations", nWifi);
  cmd.AddValue ("pcapPath", "Save a PCAP file from the AP", pcapPath);
  cmd.AddValue ("simulationTime", "Duration of the simulation excluding warmup stage (s)", simulationTime);
  cmd.AddValue ("velocity", "Velocity of the AP on X axis (m/s) [Distance mobility type]", velocity);
  cmd.AddValue ("warmupTime", "Duration of the warmup stage (s)", warmupTime);
  cmd.AddValue ("wifiManager", "Rate adaptation manager", wifiManager);
  cmd.AddValue ("wifiManagerName", "Name of the Wi-Fi manager in CSV", wifiManagerName);
  cmd.Parse (argc, argv);

  // Print simulation settings on the screen
  std::cout << std::endl
            << "Simulating an IEEE 802.11ax devices with the following settings:" << std::endl
            << "- frequency band: 5 GHz" << std::endl
            << "- loss model: " << lossModel << std::endl
            << "- delta of power changes: " << deltaPower << " dBm" << std::endl
            << "- mean interval of power changes: " << intervalPower << " s" << std::endl
            << "- max aggregated data rate: " << dataRate << " Mb/s" << std::endl
            << "- channel width: " << channelWidth << " Mhz" << std::endl
            << "- shortest guard interval: " << minGI << " ns" << std::endl
            << "- number of transmitting stations: " << nWifi << std::endl
            << "- packet size: " << packetSize << " B" << std::endl
            << "- rate adaptation manager: " << wifiManager << std::endl
            << "- simulation time: " << simulationTime << " s" << std::endl
            << "- warmup time: " << warmupTime << " s" << std::endl;

  if (mobilityModel == "Distance")
    {
      std::cout << "- mobility model: " << mobilityModel << std::endl
                << "- initial AP position: " << initialPosition << " m" << std::endl
                << "- AP velocity: " << velocity << " m/s" << std::endl
                << std::endl;
    }
  else if (mobilityModel == "RWPM")
    {
      std::cout << "- mobility model: " << mobilityModel << std::endl
                << "- area: " << area << " m" << std::endl
                << "- max node speed: " << nodeSpeed << " m/s" << std::endl
                << "- max node pause: " << nodePause << " s" << std::endl
                << std::endl;
    }

  if (wifiManagerName.empty ())
    {
      wifiManagerName = wifiManager;
    }

  // Create AP and stations
  NodeContainer wifiApNode (1);
  NodeContainer wifiStaNodes (nWifi);

  // Configure mobility
  MobilityHelper mobility;

  if (mobilityModel == "Distance")
    {
      mobility.SetMobilityModel ("ns3::ConstantVelocityMobilityModel");

      // Place all stations at (0, 0, 0)
      mobility.Install (wifiStaNodes);

      // Place AP at (initialPosition, 0, 0) and schedule start of the movement
      mobility.Install (wifiApNode);
      wifiApNode.Get (0)->GetObject<MobilityModel> ()->SetPosition (Vector3D (initialPosition, 0., 0.));

      Simulator::Schedule (Seconds (warmupTime), &StartMovement, wifiApNode.Get (0), velocity);
    }
  else if (mobilityModel == "RWPM")
    {
      // Place AP at (0, 0, 0)
      mobility.SetMobilityModel ("ns3::ConstantVelocityMobilityModel");
      mobility.Install (wifiApNode);

      // Place nodes randomly in square extending from (0, 0, 0) to (area, area, 0)
      ObjectFactory pos;
      pos.SetTypeId ("ns3::RandomRectanglePositionAllocator");
      std::stringstream ssArea;
      ssArea << "ns3::UniformRandomVariable[Min=0.0|Max=" << area;
      pos.Set ("X", StringValue (ssArea.str () + "|Stream=1]"));
      pos.Set ("Y", StringValue (ssArea.str () + "|Stream=2]"));

      Ptr<PositionAllocator> taPositionAlloc = pos.Create ()->GetObject<PositionAllocator> ();
      mobility.SetPositionAllocator (taPositionAlloc);

      // Set random pause (from 0 to nodePause) and speed (from 0 to nodeSpeed)
      std::stringstream ssSpeed;
      ssSpeed << "ns3::UniformRandomVariable[Min=0.0|Max=" << nodeSpeed << "|Stream=3]";
      std::stringstream ssPause;
      ssPause << "ns3::UniformRandomVariable[Min=0.0|Max=" << nodePause << "|Stream=4]";

      mobility.SetMobilityModel ("ns3::RandomWaypointMobilityModel",
                                 "Speed", StringValue (ssSpeed.str ()),
                                 "Pause", StringValue (ssPause.str ()),
                                 "PositionAllocator", PointerValue (taPositionAlloc));

      mobility.Install (wifiStaNodes);
    }
  else
    {
      std::cerr << "Selected incorrect mobility model!";
      return 2;
    }

  // Print position of each node
  std::cout << "Node positions:" << std::endl;

  Ptr<MobilityModel> position = wifiApNode.Get (0)->GetObject<MobilityModel> ();
  Vector3D pos = position->GetPosition ();
  std::cout << "AP:\tx=" << pos.x << ", y=" << pos.y << std::endl;

  for (auto node = wifiStaNodes.Begin (); node != wifiStaNodes.End (); ++node)
    {
      position = (*node)->GetObject<MobilityModel> ();
      pos = position->GetPosition ();
      std::cout << "Sta " << (*node)->GetId () << ":\tx=" << pos.x << ", y=" << pos.y << std::endl;
    }

  std::cout << std::endl;

  // Configure wireless channel
  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();

  if (lossModel == "Nakagami")
    {
      // Add Nakagami fading to the default log-distance model
      channel.AddPropagationLoss ("ns3::NakagamiPropagationLossModel");
    }
  else if (lossModel != "LogDistance")
    {
      std::cerr << "Selected incorrect loss model!";
      return 1;
    }

  phy.SetChannel (channel.Create ());

  // Configure two power levels
  phy.Set ("TxPowerLevels", UintegerValue (2));
  phy.Set ("TxPowerStart", DoubleValue (defaultPower - deltaPower));
  phy.Set ("TxPowerEnd", DoubleValue (defaultPower));

  // Configure MAC layer
  WifiMacHelper mac;
  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ax);

  if (wifiManager == "ns3::RLibWifiManager")
    {
      wifi.SetRemoteStationManager (wifiManager, "nWifi", UintegerValue (nWifi), "NSS", UintegerValue (nss));
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

  mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid),
               "MaxMissedBeacons", UintegerValue (1000)); // prevents exhaustion of association IDs

  NetDeviceContainer staDevices;
  staDevices = wifi.Install (phy, mac, wifiStaNodes);

  // Set channel width and shortest GI
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelSettings",
               StringValue ("{0, " + std::to_string (channelWidth) + ", BAND_5GHZ, 0}"));

  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HeConfiguration/GuardInterval",
               TimeValue (NanoSeconds (minGI)));

  // Install an Internet stack and configure IP addressing
  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiStaNodes);

  Ipv4AddressHelper address ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staNodesInterface = address.Assign (staDevices);
  Ipv4InterfaceContainer apNodeInterface = address.Assign (apDevice);

  // Populate neighbor caches for all devices
  NeighborCacheHelper neighborCache;
  neighborCache.SetDynamicNeighborCache(true);
  neighborCache.PopulateNeighborCache();

  // Configure applications
  DataRate applicationsDataRate (dataRate * 1e6 / nWifi);
  uint32_t portNumber = 9;

  for (uint32_t j = 0; j < nWifi; ++j)
    {
      InstallTrafficGenerator (wifiStaNodes.Get (j), wifiApNode.Get (0), portNumber++,
                               applicationsDataRate, packetSize, simulationTime, warmupTime);
    }

  // Install FlowMonitor
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll ();

  // Generate PCAP at AP
  if (!pcapPath.empty ())
    {
      phy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
      phy.EnablePcap (pcapPath, apDevice);
    }

  // Register callbacks for the CW and power change
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/CwTrace",
                   MakeCallback (CwCallback));

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
                   MakeCallback (PowerCallback));

  // Schedule all power changes
  double time = warmupTime;
  bool maxPower = false;

  // Create random number generator with parameters: seed = global seed, stream = 2^63 + 5, substream/run = 1
  // generator assures that the same sequence of random numbers is generated for each run
  RngStream rng (RngSeedManager::GetSeed (), ((1ULL) << 63) + 5, 1);

  while (time < warmupTime + simulationTime)
    {
      // Draw from the exponential distribution ( [ -1 / lambda * ln(x) ] ~ Exp[lambda] where x ~ Uniform[0, 1] )
      time += -intervalPower * std::log (rng.RandU01 ());

      // The interval between each change follows the exponential distribution
      Simulator::Schedule (Seconds (time), &ChangePower, wifiStaNodes, maxPower);
      maxPower = !maxPower;
    }

  // Setup CSV output
  std::ostringstream csvPrefixStream;
  csvPrefixStream << wifiManagerName << ',' << lossModel << ',' << mobilityModel << ',' << channelWidth << ','
                  << minGI << ',' << (mobilityModel == "Distance" ? velocity : nodeSpeed) << ',' << deltaPower << ','
                  << intervalPower << ',' << RngSeedManager::GetRun () << ',' << nWifi << ",{nWifiReal}";

  csvPrefix = csvPrefixStream.str ();
  std::string csvHeader = "wifiManager,lossModel,mobilityModel,channelWidth,minGI,velocity,"
                          "delta,interval,seed,nWifi,nWifiReal,position,time,throughput\n";

  // Schedule measurements
  Simulator::Schedule (Seconds (warmupTime), &WarmupMeasurement, monitor);
  Simulator::Schedule (Seconds (warmupTime), &Measurement, monitor, wifiApNode.Get (0), warmupTime, logEvery);

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

  // Calculate per-flow throughput and Jain's fairness index
  uint32_t nWifiReal = 0;
  double jainsIndexN = 0.;
  double jainsIndexD = 0.;

  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
  std::cout << "Results: " << std::endl;

  for (const auto &stat : flowState.staFlows)
    {
      double stationThroughput = (stat.second - warmupState.staFlows[stat.first]) / (1e6 * simulationTime);

      if (stationThroughput > 0)
        {
          nWifiReal += 1;
        }

      jainsIndexN += stationThroughput;
      jainsIndexD += stationThroughput * stationThroughput;

      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (stat.first);
      std::cout << "Flow " << stat.first << " (" << t.sourceAddress << " -> "
                << t.destinationAddress << ")\tThroughput: " << stationThroughput << " Mb/s"
                << std::endl;
    }

  double totalThr = jainsIndexN;
  double fairnessIndex = jainsIndexN * jainsIndexN / (nWifi * jainsIndexD);

  // Print results
  std::cout << std::endl
            << "Network throughput: " << totalThr << " Mb/s" << std::endl
            << "Jain's fairness index: " << fairnessIndex << std::endl
            << std::endl;

  std::string csvString = csvOutput.str ();
  csvString = std::regex_replace (csvString, std::regex ("\\{nWifiReal\\}"), std::to_string (nWifiReal));

  std::cout << csvHeader << csvString;

  // Save file in CSV format
  if (!csvPath.empty ())
    {
      std::ofstream csvFile (csvPath);
      csvFile << csvString;
    }

  //Clean-up
  Simulator::Destroy ();
  m_env->SetFinish ();

  return 0;
}

/***** Function definitions *****/

void
ChangePower (NodeContainer wifiStaNodes, uint8_t powerLevel)
{
  // Iter through STA nodes and change power for each node
  for (auto node = wifiStaNodes.Begin (); node != wifiStaNodes.End (); ++node)
    {
      Config::Set ("/NodeList/" + std::to_string ((*node)->GetId ()) + "/DeviceList/*/$ns3::WifiNetDevice/"
                       "RemoteStationManager/DefaultTxPowerLevel",
                   UintegerValue (powerLevel));
    }
}

void
CwCallback (std::string path, u_int32_t value, u_int8_t linkId)
{
  size_t start = 10; // the length of the "/NodeList/" string
  size_t end = path.find ("/DeviceList/");
  std::string nodeId = path.substr (start, end - start);

  // Update CW value in the RLibWifiManager manager instance
  Config::Set ("/NodeList/" + nodeId + "/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
                   "$ns3::RLibWifiManager/CW",
               UintegerValue (value));
}

void
InstallTrafficGenerator (Ptr<ns3::Node> sourceNode, Ptr<ns3::Node> sinkNode, uint32_t port,
                         DataRate offeredLoad, uint32_t packetSize, double simulationTime,
                         double warmupTime)
{
  // Get sink address
  Ptr<Ipv4> ipv4 = sinkNode->GetObject<Ipv4> ();
  Ipv4Address addr = ipv4->GetAddress (1, 0).GetLocal ();

  // Define type of service
  uint8_t tosValue = 0x70; //AC_BE

  // Add random fuzz to app start time
  Ptr<UniformRandomVariable> fuzz = CreateObject<UniformRandomVariable> ();
  fuzz->SetAttribute ("Min", DoubleValue (0.));
  fuzz->SetAttribute ("Max", DoubleValue (warmupTime / 2));
  fuzz->SetStream (6);
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
Measurement (Ptr<FlowMonitor> monitor, Ptr<Node> sinkNode, double warmupTime, double nextMeasurement)
{
  // Initial metrics values
  static double lastTime = -1.;
  static uint64_t lastReceivedBits = warmupState.receivedBits;
  
  if (lastTime == -1.)
    {
      lastTime = Simulator::Now ().GetSeconds ();
      Simulator::Schedule (Seconds (nextMeasurement), &Measurement, monitor, sinkNode, warmupTime, nextMeasurement);
      return;
    }

  // Calculate metrics since the previous measurement
  double currentTime = Simulator::Now ().GetSeconds ();
  uint64_t receivedBits = 0;

  for (auto &stat : monitor->GetFlowStats ())
    {
      uint64_t bits = 8 * stat.second.rxBytes;
      receivedBits += bits;
      flowState.staFlows[stat.first] = bits;
    }

  double throughput = (receivedBits - lastReceivedBits) / (1e6 * (currentTime - lastTime));
  double position = sinkNode->GetObject<MobilityModel> ()->GetPosition ().x;

  lastTime = currentTime;
  lastReceivedBits = receivedBits;
  flowState.receivedBits = receivedBits;

  // Add current state to CSV
  csvOutput << csvPrefix << ',' << position << ',' << currentTime - warmupTime << ',' << throughput << std::endl;

  // Schedule next measurement
  Simulator::Schedule (Seconds (nextMeasurement), &Measurement, monitor, sinkNode, warmupTime, nextMeasurement);
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
StartMovement (Ptr<Node> node, double velocity)
{
  node->GetObject<ConstantVelocityMobilityModel> ()->SetVelocity (Vector3D (velocity, 0., 0.));
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
}
