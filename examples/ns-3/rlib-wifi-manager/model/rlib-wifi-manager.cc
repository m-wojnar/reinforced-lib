#include "ns3/string.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "rlib-wifi-manager.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-tx-vector.h"
#include "ns3/wifi-utils.h"

#define Min(a,b) ((a < b) ? a : b)
#define DEFAULT_MEMBLOCK_KEY 2333

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RLibWifiManager");

NS_OBJECT_ENSURE_REGISTERED (RLibWifiManager);

struct RLibWifiRemoteStation : public WifiRemoteStation
{
  uint32_t m_station_id;
};

TypeId
RLibWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RLibWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<RLibWifiManager> ()
    .AddAttribute ("DataMode", "The transmission mode to use for every data packet transmission",
                   StringValue ("HeMcs0"),
                   MakeWifiModeAccessor (&RLibWifiManager::m_dataMode),
                   MakeWifiModeChecker ())
    .AddAttribute ("ControlMode", "The transmission mode to use for every RTS packet transmission.",
                   StringValue ("OfdmRate6Mbps"),
                   MakeWifiModeAccessor (&RLibWifiManager::m_ctlMode),
                   MakeWifiModeChecker ())
    .AddAttribute ("CW", "Current contention window",
                   UintegerValue (15),
                   MakeUintegerAccessor (&RLibWifiManager::m_cw),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("nWifi", "Number of Wifi stations",
                   UintegerValue (1),
                   MakeUintegerAccessor (&RLibWifiManager::m_nWifi),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Power", "Current transmission power [dBm]",
                   DoubleValue (16.0206),
                   MakeDoubleAccessor (&RLibWifiManager::m_power),
                   MakeDoubleChecker<double_t> ())
  ;
  return tid;
}

RLibWifiManager::RLibWifiManager ()
{
  NS_LOG_FUNCTION (this);

  // Setup ns3-ai Env
  m_env = new Ns3AIRL<sEnv, sAct> (DEFAULT_MEMBLOCK_KEY);
  m_env->SetCond (2, 0);
}

RLibWifiManager::~RLibWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

WifiRemoteStation *
RLibWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);

  RLibWifiRemoteStation *station = new RLibWifiRemoteStation ();

  // Initialize new station
  auto env = m_env->EnvSetterCond ();
  env->type = 0;
  m_env->SetCompleted ();

  // Get station ID
  auto act = m_env->ActionGetterCond ();
  station->m_station_id = act->station_id;
  m_env->GetCompleted ();

  return station;
}

void
RLibWifiManager::DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void
RLibWifiManager::DoReportAmpduTxStatus (WifiRemoteStation *st, uint16_t nSuccessfulMpdus,
                                          uint16_t nFailedMpdus, double rxSnr, double dataSnr,
                                          uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << nSuccessfulMpdus << nFailedMpdus << rxSnr << dataSnr
                        << dataChannelWidth << dataNss);

  auto station = static_cast<RLibWifiRemoteStation *> (st);
  UpdateState (station->m_station_id, nSuccessfulMpdus, nFailedMpdus);
  ExecuteAction (station->m_station_id);
}

void
RLibWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
RLibWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);

  auto station = static_cast<RLibWifiRemoteStation *> (st);
  UpdateState (station->m_station_id, 0, 1);
  ExecuteAction (station->m_station_id);
}

void
RLibWifiManager::DoReportRtsOk (WifiRemoteStation *st, double ctsSnr, WifiMode ctsMode,
                                  double rtsSnr)
{
  NS_LOG_FUNCTION (this << st << ctsSnr << ctsMode << rtsSnr);
}

void
RLibWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                   double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);

  auto station = static_cast<RLibWifiRemoteStation *> (st);
  UpdateState (station->m_station_id, 1, 0);
  ExecuteAction (station->m_station_id);
}

void
RLibWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
RLibWifiManager::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

WifiTxVector
RLibWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);

  uint8_t nss = Min (GetMaxNumberOfTransmitStreams (), GetNumberOfSupportedStreams (st));
  if (m_dataMode.GetModulationClass () == WIFI_MOD_CLASS_HT)
    {
      nss = 1 + (m_dataMode.GetMcsValue () / 8);
    }

  return WifiTxVector (
      m_dataMode,
      GetDefaultTxPowerLevel (),
      GetPreambleForTransmission (m_dataMode.GetModulationClass (), GetShortPreambleEnabled ()),
      ConvertGuardIntervalToNanoSeconds (m_dataMode, GetShortGuardIntervalSupported (st), NanoSeconds (GetGuardInterval (st))),
      GetNumberOfAntennas (),
      nss,
      0,
      GetChannelWidthForTransmission (m_dataMode, GetChannelWidth (st)),
      GetAggregation (st));
}

WifiTxVector
RLibWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);

  return WifiTxVector (
      m_ctlMode,
      GetDefaultTxPowerLevel (),
      GetPreambleForTransmission (m_ctlMode.GetModulationClass (), GetShortPreambleEnabled ()),
      ConvertGuardIntervalToNanoSeconds (m_ctlMode, GetShortGuardIntervalSupported (st), NanoSeconds (GetGuardInterval (st))),
      1,
      1,
      0,
      GetChannelWidthForTransmission (m_ctlMode, GetChannelWidth (st)),
      GetAggregation (st));
}

void
RLibWifiManager::UpdateState (uint32_t station_id, uint16_t nSuccessful, uint16_t nFailed)
{
  // Write observation to shared memory
  auto env = m_env->EnvSetterCond ();

  env->station_id = station_id;
  env->type = 1;
  env->time = Simulator::Now ().GetSeconds ();
  env->n_successful = nSuccessful;
  env->n_failed = nFailed;
  env->n_wifi = m_nWifi;
  env->power = m_power;
  env->cw = m_cw;
  env->mcs = m_mcs;

  m_env->SetCompleted ();
}

void
RLibWifiManager::ExecuteAction (uint32_t station_id)
{
  // Get selected action
  auto act = m_env->ActionGetterCond ();

  if (act->station_id != station_id)
    {
      NS_ASSERT_MSG (act->station_id == station_id, "Error! Different station_id in ns3-ai action and environment structures!");
    }
  m_mcs = act->mode;

  m_env->GetCompleted ();

  // Set new MCS
  WifiMode mode("HeMcs" + std::to_string (m_mcs));
  m_dataMode = mode;
}

} //namespace ns3
