#ifndef RLIB_WIFI_MANAGER_H
#define RLIB_WIFI_MANAGER_H

#include "ns3/wifi-remote-station-manager.h"
#include "ns3/ns3-ai-module.h"

namespace ns3 {

// ns3-ai structures
struct sEnv
{
  double power;
  double time;
  uint32_t cw;
  uint32_t n_failed;
  uint32_t n_successful;
  uint32_t n_wifi;
  uint32_t station_id;
  uint8_t mcs;
  uint8_t type;
} Packed;

struct sAct
{
  uint32_t station_id;
  uint8_t mcs;
} Packed;

// Structure holding additional information required by the RLibWifiManager
struct RLibWifiRemoteStation : public WifiRemoteStation
{
  uint32_t m_station_id;
  uint8_t m_mcs;
};

class RLibWifiManager : public WifiRemoteStationManager
{
public:
  static TypeId GetTypeId (void);
  RLibWifiManager ();
  virtual ~RLibWifiManager ();

private:
  WifiRemoteStation *DoCreateStation (void) const override;
  void DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode) override;
  void DoReportAmpduTxStatus (WifiRemoteStation *station, uint16_t nSuccessfulMpdus,
                              uint16_t nFailedMpdus, double rxSnr, double dataSnr,
                              uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportRtsFailed (WifiRemoteStation *station) override;
  void DoReportDataFailed (WifiRemoteStation *station) override;
  void DoReportRtsOk (WifiRemoteStation *station, double ctsSnr, WifiMode ctsMode,
                      double rtsSnr) override;
  void DoReportDataOk (WifiRemoteStation *station, double ackSnr, WifiMode ackMode, double dataSnr,
                       uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportFinalRtsFailed (WifiRemoteStation *station) override;
  void DoReportFinalDataFailed (WifiRemoteStation *station) override;
  WifiTxVector DoGetDataTxVector (WifiRemoteStation *station) override;
  WifiTxVector DoGetRtsTxVector (WifiRemoteStation *station) override;

  void UpdateState (RLibWifiRemoteStation *station, uint16_t nSuccessful, uint16_t nFailed);
  void ExecuteAction (RLibWifiRemoteStation *station);

  WifiMode m_ctlMode;
  uint32_t m_cw;
  uint32_t m_nWifi;
  uint32_t m_nss;
  double m_power;

  Ns3AIRL<sEnv, sAct> * m_env;
};

}

#endif
