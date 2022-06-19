#ifndef RLIB_WIFI_MANAGER_H
#define RLIB_WIFI_MANAGER_H

#include "ns3/wifi-remote-station-manager.h"
#include "ns3/ns3-ai-module.h"

namespace ns3 {

// ns3-ai structures
struct sEnv
{
  uint32_t station_id;
  uint8_t type;
  double time;
  uint32_t n_successful;
  uint32_t n_failed;
  uint32_t n_wifi;
  double power;
  uint32_t cw;
  uint8_t mcs;
} Packed;

struct sAct
{
  uint32_t station_id;
  uint8_t mode;
} Packed;

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

  void UpdateState (uint32_t station_id, uint16_t nSuccessful, uint16_t nFailed);
  void ExecuteAction (uint32_t station_id);

  WifiMode m_ctlMode;     // Wifi mode for RTS frames
  WifiMode m_dataMode;    // Wifi mode for data frames
  uint32_t m_cw;          // current Contention Window
  uint32_t m_nWifi;       // number of transmitting stations
  double m_power;         // current transmission power
  uint8_t m_mcs;          // currently used MCS

  Ns3AIRL<sEnv, sAct> * m_env;  // ns3-ai environment
};

}

#endif
