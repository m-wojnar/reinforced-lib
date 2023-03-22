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

TypeId
RLibWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RLibWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<RLibWifiManager> ()
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
    .AddAttribute ("NSS", "Number of spatial streams",
                   UintegerValue (1),
                   MakeUintegerAccessor(&RLibWifiManager::m_nss),
                   MakeUintegerChecker<uint32_t> ())
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
  station->m_mcs = 0;

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
  UpdateState (station, nSuccessfulMpdus, nFailedMpdus);
  ExecuteAction (station);
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
  UpdateState (station, 0, 1);
  ExecuteAction (station);
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
  UpdateState (station, 1, 0);
  ExecuteAction (station);
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
RLibWifiManager::DoGetDataTxVector (WifiRemoteStation *st, uint16_t allowedWidth)
{
  NS_LOG_FUNCTION (this << st);

  auto station = static_cast<RLibWifiRemoteStation *> (st);
  WifiMode dataMode ("HeMcs" + std::to_string (station->m_mcs));
  uint16_t channelWidth = std::min (allowedWidth, GetChannelWidthForTransmission (dataMode, GetChannelWidth (st)));

  return WifiTxVector (
      dataMode,
      GetDefaultTxPowerLevel (),
      GetPreambleForTransmission (dataMode.GetModulationClass (), GetShortPreambleEnabled ()),
      ConvertGuardIntervalToNanoSeconds (dataMode, GetShortGuardIntervalSupported (st), NanoSeconds (GetGuardInterval (st))),
      GetNumberOfAntennas (),
      m_nss,
      0,
      channelWidth,
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
      GetNumberOfAntennas (),
      m_nss,
      0,
      GetChannelWidthForTransmission (m_ctlMode, GetChannelWidth (st)),
      GetAggregation (st));
}

void
RLibWifiManager::UpdateState (RLibWifiRemoteStation *st, uint16_t nSuccessful, uint16_t nFailed)
{
  // Write observation to shared memory
  auto env = m_env->EnvSetterCond ();

  env->power = m_power;
  env->time = Simulator::Now ().GetSeconds ();
  env->cw = m_cw;
  env->n_failed = nFailed;
  env->n_successful = nSuccessful;
  env->n_wifi = m_nWifi;
  env->station_id = st->m_station_id;
  env->type = 1;

  m_env->SetCompleted ();
}

void
RLibWifiManager::ExecuteAction (RLibWifiRemoteStation *st)
{
  // Get selected action
  auto act = m_env->ActionGetterCond ();

  if (act->station_id != st->m_station_id)
    {
      NS_ASSERT_MSG (act->station_id == st->m_station_id, "Error! Different station_id in ns3-ai action and environment structures!");
    }
  st->m_mcs = act->mcs;

  m_env->GetCompleted ();
}

}
