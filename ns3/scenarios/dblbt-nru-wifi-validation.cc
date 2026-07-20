/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/applications-module.h"
#include "ns3/beamforming-vector.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/mobility-module.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "../model/nr-db-lbt-access-manager.h"
#include "ns3/nr-module.h"
#include "ns3/nr-u-module.h"
#include "ns3/non-communicating-net-device.h"
#include "ns3/sqlite-output.h"
#include "ns3/spectrum-module.h"
#include "ns3/three-gpp-spectrum-propagation-loss-model.h"
#include "ns3/uniform-planar-array.h"
#include "ns3/waveform-generator-helper.h"
#include "ns3/wifi-module.h"
#include "ns3/wifi-spectrum-value-helper.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("DbLbtNruWifiValidation");

namespace {

constexpr double kFrequency = 5.2e9;
constexpr double kBandwidth = 20e6;
constexpr uint32_t kPacketSize = 1000;
const std::string kModelHash =
  "70611e9712e3a8e4b0f35fc8e0e616f4fde9a20b17c844f1b6406da2833301e6";
const std::string kGridHash =
  "558da7340dfa32d8cc484ba68a05951314936d7aff34a145cc34ea051c07707c";
const std::string kTrafficMode = "aggregate-saturated-cbr";
const std::string kNs3Commit = "ac88b75eac1818c673cf2c939a96ac3005b1f051";
const std::string kNrCommit = "fe0a1d2a5fb7d1547e46042041288a684893ba9e";
const std::string kNruCommit = "75a45143b1cd382326876a9597e856338673039a";

struct OccupancyTracker
{
  Time end {Seconds (0)};
  Time total {Seconds (0)};

  bool Active () const
  {
    return end > Simulator::Now ();
  }

  void Observe (Time duration)
  {
    Time now = Simulator::Now ();
    Time nextEnd = now + duration;
    if (now >= end)
      {
        total += duration;
      }
    else if (nextEnd > end)
      {
        total += nextEnd - end;
      }
    end = std::max (end, nextEnd);
  }
};

enum class RadioKind
{
  Wifi,
  Nru
};

struct ControllerBinding
{
  uint32_t nodeId {0};
  std::string technology;
  std::string stateId;
  Ptr<DbLbtLocalController> controller;
};

struct TechnologyMetrics
{
  double throughputMbps {0.0};
  double meanDelayUs {0.0};
  double collisionProbability {0.0};
  double channelOccupancy {0.0};
};

SqliteOutputManager *g_output = nullptr;
OccupancyTracker g_wifiOccupancy;
OccupancyTracker g_nruOccupancy;

std::string
Quote (const std::string &value)
{
  std::string escaped;
  escaped.reserve (value.size () + 2);
  escaped.push_back ('\'');
  for (char character : value)
    {
      escaped.push_back (character);
      if (character == '\'')
        {
          escaped.push_back ('\'');
        }
    }
  escaped.push_back ('\'');
  return escaped;
}

std::string
Number (double value)
{
  NS_ABORT_MSG_IF (!std::isfinite (value), "refusing to store non-finite metric");
  std::ostringstream output;
  output << std::setprecision (17) << value;
  return output.str ();
}

void
Exec (SQLiteOutput &database, const std::string &statement)
{
  NS_ABORT_MSG_IF (!database.WaitExec (statement),
                   "SQLite statement failed: " << statement);
}

bool
IsHash (const std::string &value)
{
  return value.size () == 64 &&
         std::all_of (value.begin (), value.end (), [] (char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void
ConfigureDefaults ()
{
  Config::SetDefault ("ns3::ThreeGppPropagationLossModel::ShadowingEnabled",
                      BooleanValue (false));
  Config::SetDefault ("ns3::IdealBeamformingHelper::BeamformingMethod",
                      TypeIdValue (CellScanBeamforming::GetTypeId ()));
  Config::SetDefault ("ns3::IdealBeamformingHelper::BeamformingPeriodicity",
                      TimeValue (Seconds (1)));
  Config::SetDefault ("ns3::CellScanBeamforming::BeamSearchAngleStep",
                      DoubleValue (30.0));
  Config::SetDefault ("ns3::UniformPlanarArray::NumColumns", UintegerValue (2));
  Config::SetDefault ("ns3::UniformPlanarArray::NumRows", UintegerValue (2));
  Config::SetDefault ("ns3::NrGnbPhy::NoiseFigure", DoubleValue (7));
  Config::SetDefault ("ns3::NrUePhy::NoiseFigure", DoubleValue (7));
  Config::SetDefault ("ns3::WifiPhy::RxNoiseFigure", DoubleValue (7));
  Config::SetDefault ("ns3::NrSpectrumPhy::UnlicensedMode", BooleanValue (true));
  Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (999999999));
  Config::SetDefault ("ns3::PointToPointEpcHelper::S1uLinkDelay",
                      TimeValue (Seconds (0)));
  Config::SetDefault ("ns3::NoBackhaulEpcHelper::X2LinkDelay",
                      TimeValue (Seconds (0)));
  Config::SetDefault ("ns3::LteEnbRrc::EpsBearerToRlcMapping",
                      StringValue ("RlcUmAlways"));
  Config::SetDefault ("ns3::NrAmc::ErrorModelType",
                      TypeIdValue (NrEesmIrT1::GetTypeId ()));
  Config::SetDefault ("ns3::NrAmc::AmcModel", EnumValue (NrAmc::ShannonModel));
  Config::SetDefault ("ns3::NrSpectrumPhy::ErrorModelType",
                      TypeIdValue (NrEesmIrT1::GetTypeId ()));
  Config::SetDefault ("ns3::NrMacSchedulerNs3::EnableSrsInFSlots",
                      BooleanValue (false));
  Config::SetDefault ("ns3::NrMacSchedulerNs3::EnableSrsInUlSlots",
                      BooleanValue (false));
  Config::SetDefault ("ns3::WifiRemoteStationManager::FragmentationThreshold",
                      StringValue ("999999"));
  Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold",
                      StringValue ("999999"));
  Config::SetDefault ("ns3::ApWifiMac::EnableBeaconJitter", BooleanValue (true));
  Config::SetDefault ("ns3::NrLbtAccessManager::EnergyDetectionThreshold",
                      DoubleValue (-79.0));
}

void
InstallPositions (const NodeContainer &wifiAps,
                  const NodeContainer &wifiStas,
                  const NodeContainer &nruGnbs,
                  const NodeContainer &nruUes)
{
  NodeContainer all;
  all.Add (wifiAps);
  all.Add (wifiStas);
  all.Add (nruGnbs);
  all.Add (nruUes);
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (all);
  for (uint32_t index = 0; index < wifiAps.GetN (); ++index)
    {
      double y = 2.0 + 3.0 * index;
      wifiAps.Get (index)->GetObject<MobilityModel> ()->SetPosition ({1.0, y, 1.5});
      wifiStas.Get (index)->GetObject<MobilityModel> ()->SetPosition ({3.0, y, 1.0});
    }
  for (uint32_t index = 0; index < nruGnbs.GetN (); ++index)
    {
      double y = 2.0 + 3.0 * index;
      nruGnbs.Get (index)->GetObject<MobilityModel> ()->SetPosition ({9.0, y, 1.5});
      nruUes.Get (index)->GetObject<MobilityModel> ()->SetPosition ({7.0, y, 1.0});
    }
}

void
ObserveOccupancy (RadioKind kind, uint32_t nodeId, const Time &duration)
{
  OccupancyTracker &own = kind == RadioKind::Wifi ? g_wifiOccupancy : g_nruOccupancy;
  OccupancyTracker &other = kind == RadioKind::Wifi ? g_nruOccupancy : g_wifiOccupancy;
  g_output->UidIsTxing (nodeId);
  if (own.Active ())
    {
      g_output->SimultaneousTxSameTechnology (nodeId);
    }
  if (other.Active ())
    {
      g_output->SimultaneousTxOtherTechnology (nodeId);
    }
  own.Observe (duration);
}

uint32_t
WifiBackoff (Ptr<DbLbtLocalController> controller,
             Ptr<QosTxop> txop,
             uint32_t currentCw)
{
  Ptr<WifiMacQueue> queue = txop->GetWifiMacQueue ();
  double maximum = static_cast<double> (queue->GetMaxSize ().GetValue ());
  double occupancy = maximum > 0.0 ? queue->GetNPackets () / maximum : 0.0;
  controller->SetQueueOccupancy (occupancy);
  return controller->NextBackoff (currentCw);
}

void
WifiPhyState (Ptr<DbLbtLocalController> controller,
              Time start,
              Time duration,
              WifiPhyState state)
{
  NS_UNUSED (start);
  if (state == WifiPhyState::TX)
    {
      controller->NotifyOwnTx (duration);
    }
}

Ptr<DbLbtLocalController>
ConfigureController (const std::string &policy,
                     const std::string &modelPath,
                     const std::string &modelHash,
                     const std::string &gridHash,
                     int64_t stream)
{
  Ptr<DbLbtLocalController> controller = CreateObject<DbLbtLocalController> ();
  if (policy == "adaptive")
    {
      controller->ConfigureAdaptive (modelPath, modelHash, gridHash);
    }
  else
    {
      controller->ConfigureTmc ();
    }
  controller->AssignStreams (stream);
  return controller;
}

void
BindWifiController (Ptr<WifiNetDevice> device,
                    Ptr<DbLbtLocalController> controller)
{
  Ptr<RegularWifiMac> mac = DynamicCast<RegularWifiMac> (device->GetMac ());
  NS_ABORT_MSG_IF (!mac, "Wi-Fi device lacks RegularWifiMac");
  Ptr<QosTxop> txop = mac->GetQosTxop (AC_BE);
  txop->SetDbLbtBackoffCallback (
    MakeBoundCallback (&WifiBackoff, controller, txop));
  txop->SetDbLbtBusyCallback (
    MakeCallback (&DbLbtLocalController::NotifyBusy, controller));
  txop->SetDbLbtOutcomeCallback (
    MakeCallback (&DbLbtLocalController::NotifyOutcome, controller));
  txop->SetDbLbtGrantCallback (
    MakeCallback (&DbLbtLocalController::NotifyGrant, controller));

  Ptr<SpectrumWifiPhy> phy = DynamicCast<SpectrumWifiPhy> (device->GetPhy ());
  PointerValue stateValue;
  phy->GetAttribute ("State", stateValue);
  Ptr<WifiPhyStateHelper> state =
    DynamicCast<WifiPhyStateHelper> (stateValue.Get<WifiPhyStateHelper> ());
  state->TraceConnectWithoutContext (
    "State", MakeBoundCallback (&WifiPhyState, controller));
}

void
InstallInterferer (Ptr<MultiModelSpectrumChannel> channel,
                   Ptr<ThreeGppSpectrumPropagationLossModel> spectrumPropagation,
                   double appStart,
                   double simTime,
                   uint32_t interferenceIntervalMs,
                   uint32_t interferenceDurationMs)
{
  NS_ABORT_MSG_IF (interferenceIntervalMs != 300 ||
                     interferenceDurationMs == 0 ||
                     interferenceDurationMs >= interferenceIntervalMs,
                   "non-ideal scenario requires a 300 ms finite interference interval");
  NodeContainer node;
  node.Create (1);
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (node);
  node.Get (0)->GetObject<MobilityModel> ()->SetPosition ({5.0, 5.0, 1.5});
  Ptr<SpectrumValue> psd =
    WifiSpectrumValueHelper::CreateOfdmTxPowerSpectralDensity (
      5200, 20, 0.01, 20);
  WaveformGeneratorHelper helper;
  helper.SetChannel (channel);
  helper.SetTxPowerSpectralDensity (psd);
  helper.SetPhyAttribute ("Period",
                          TimeValue (MilliSeconds (interferenceIntervalMs)));
  helper.SetPhyAttribute (
    "DutyCycle",
    DoubleValue (static_cast<double> (interferenceDurationMs) /
                 interferenceIntervalMs));
  NetDeviceContainer devices = helper.Install (node);
  Ptr<PhasedArrayModel> interferenceAntenna =
    CreateObjectWithAttributes<UniformPlanarArray> (
      "NumColumns", UintegerValue (1),
      "NumRows", UintegerValue (1),
      "AntennaElement", PointerValue (CreateObject<IsotropicAntennaModel> ()));
  spectrumPropagation->AddDevice (devices.Get (0), interferenceAntenna);
  Ptr<WaveformGenerator> generator =
    devices.Get (0)->GetObject<NonCommunicatingNetDevice> ()
      ->GetPhy ()->GetObject<WaveformGenerator> ();
  Simulator::Schedule (Seconds (appStart), &WaveformGenerator::Start, generator);
  Simulator::Schedule (Seconds (simTime), &WaveformGenerator::Stop, generator);
}

void
CreateValidationTables (SQLiteOutput &database)
{
  Exec (database,
        "CREATE TABLE IF NOT EXISTS validation_metadata ("
        "schema_version INTEGER NOT NULL, job_id TEXT NOT NULL, "
        "policy TEXT NOT NULL, scenario TEXT NOT NULL, seed INTEGER NOT NULL, "
        "run_id INTEGER NOT NULL, wifi_aps INTEGER NOT NULL, "
        "nru_gnbs INTEGER NOT NULL, node_rate_bps INTEGER NOT NULL, "
        "traffic_mode TEXT NOT NULL, srs_enabled INTEGER NOT NULL, "
        "alpha INTEGER NOT NULL, "
        "cold_start_attempts INTEGER NOT NULL, decision_interval INTEGER NOT NULL, "
        "context_dim INTEGER NOT NULL, num_arms INTEGER NOT NULL, "
        "model_sha256 TEXT NOT NULL, model_export_sha256 TEXT NOT NULL, "
        "action_grid_hash TEXT NOT NULL, ns3_commit TEXT NOT NULL, "
        "nr_commit TEXT NOT NULL, nru_commit TEXT NOT NULL, "
        "patch_sha256 TEXT NOT NULL, scenario_sha256 TEXT NOT NULL);");
  Exec (database,
        "CREATE TABLE IF NOT EXISTS dblbt_nodes (node_id INTEGER PRIMARY KEY, "
        "technology TEXT NOT NULL, state_id TEXT NOT NULL UNIQUE);");
  Exec (database,
        "CREATE TABLE IF NOT EXISTS dblbt_attempts (node_id INTEGER NOT NULL, "
        "attempt_id INTEGER NOT NULL, outcome TEXT NOT NULL, elapsed_us REAL NOT NULL, "
        "busy_us REAL NOT NULL, interruptions INTEGER NOT NULL, "
        "access_delay_us REAL NOT NULL, queue_occupancy REAL NOT NULL, "
        "arrivals INTEGER NOT NULL, retries INTEGER NOT NULL, "
        "effective_data_us REAL NOT NULL, selected_backoff INTEGER NOT NULL, "
        "PRIMARY KEY (node_id, attempt_id));");
  Exec (database,
        "CREATE TABLE IF NOT EXISTS dblbt_decisions (node_id INTEGER NOT NULL, "
        "decision_round INTEGER NOT NULL, arm_id INTEGER NOT NULL, "
        "kappa INTEGER NOT NULL, alpha INTEGER NOT NULL, beta INTEGER NOT NULL, "
        "m INTEGER NOT NULL, b_init INTEGER NOT NULL, reward REAL NOT NULL, "
        "context_0 REAL NOT NULL, context_1 REAL NOT NULL, context_2 REAL NOT NULL, "
        "context_3 REAL NOT NULL, context_4 REAL NOT NULL, context_5 REAL NOT NULL, "
        "context_6 REAL NOT NULL, context_7 REAL NOT NULL, context_8 REAL NOT NULL, "
        "context_9 REAL NOT NULL, context_10 REAL NOT NULL, "
        "PRIMARY KEY (node_id, decision_round));");
  Exec (database,
        "CREATE TABLE IF NOT EXISTS validation_metrics (technology TEXT PRIMARY KEY, "
        "throughput_mbps REAL NOT NULL, mean_delay_us REAL NOT NULL, "
        "collision_probability REAL NOT NULL, channel_occupancy REAL NOT NULL);");
  for (const std::string table : {"validation_metadata", "dblbt_nodes",
                                  "dblbt_attempts", "dblbt_decisions",
                                  "validation_metrics"})
    {
      Exec (database, "DELETE FROM " + table + ";");
    }
}

void
WriteValidationOutput (
  const std::string &outputDb,
  const std::string &jobId,
  const std::string &policy,
  const std::string &scenario,
  uint32_t seed,
  uint32_t runId,
  uint32_t wifiAps,
  uint32_t nruGnbs,
  uint64_t nodeRateBps,
  const std::string &modelHash,
  const std::string &modelExportHash,
  const std::string &gridHash,
  const std::string &patchHash,
  const std::string &scenarioHash,
  const std::vector<ControllerBinding> &bindings,
  const std::map<std::string, TechnologyMetrics> &metrics)
{
  SQLiteOutput database (outputDb, "/dblbt-" + jobId + "-custom");
  CreateValidationTables (database);
  Exec (database,
        "INSERT INTO validation_metadata VALUES (1," + Quote (jobId) + "," +
        Quote (policy) + "," + Quote (scenario) + "," + std::to_string (seed) +
        "," + std::to_string (runId) + "," + std::to_string (wifiAps) + "," +
        std::to_string (nruGnbs) + "," + std::to_string (nodeRateBps) + "," +
        Quote (kTrafficMode) + ",0,11,8,32,11,24," + Quote (modelHash) +
        "," + Quote (modelExportHash) + "," + Quote (gridHash) + "," +
        Quote (kNs3Commit) + "," + Quote (kNrCommit) + "," + Quote (kNruCommit) +
        "," + Quote (patchHash) + "," + Quote (scenarioHash) + ");");

  for (const auto &binding : bindings)
    {
      Exec (database,
            "INSERT INTO dblbt_nodes VALUES (" +
            std::to_string (binding.nodeId) + "," + Quote (binding.technology) +
            "," + Quote (binding.stateId) + ");");
      if (!binding.controller)
        {
          continue;
        }
      uint64_t attemptId = 0;
      for (const auto &attempt : binding.controller->GetAttempts ())
        {
          ++attemptId;
          Exec (database,
                "INSERT INTO dblbt_attempts VALUES (" +
                std::to_string (binding.nodeId) + "," + std::to_string (attemptId) +
                "," + Quote (attempt.success ? "success" : "collision") + "," +
                Number (attempt.elapsedUs) + "," + Number (attempt.busyUs) + "," +
                std::to_string (attempt.interruptions) + "," +
                Number (attempt.accessDelayUs) + "," +
                Number (attempt.queueOccupancy) + "," +
                std::to_string (attempt.arrivals) + "," +
                std::to_string (attempt.retries) + "," +
                Number (attempt.effectiveDataUs) + "," +
                std::to_string (attempt.selectedBackoff) + ");");
        }
      for (const auto &decision : binding.controller->GetDecisions ())
        {
          std::string statement =
            "INSERT INTO dblbt_decisions VALUES (" +
            std::to_string (binding.nodeId) + "," +
            std::to_string (decision.attemptId) + "," +
            std::to_string (decision.armId) + "," +
            std::to_string (decision.profile.kappa) + "," +
            std::to_string (decision.profile.alpha) + "," +
            std::to_string (decision.profile.beta) + "," +
            std::to_string (decision.profile.m) + "," +
            std::to_string (decision.profile.bInit) + "," +
            Number (decision.reward);
          for (double value : decision.context)
            {
              statement += "," + Number (value);
            }
          Exec (database, statement + ");");
        }
    }

  for (const std::string technology : {"wifi", "nru"})
    {
      const auto &value = metrics.at (technology);
      Exec (database,
            "INSERT INTO validation_metrics VALUES (" + Quote (technology) +
            "," + Number (value.throughputMbps) + "," +
            Number (value.meanDelayUs) + "," +
            Number (value.collisionProbability) + "," +
            Number (value.channelOccupancy) + ");");
    }
}

} // namespace

int
main (int argc, char *argv[])
{
  std::string policy = "tmc";
  std::string scenario = "static-4x4";
  std::string modelPath;
  std::string modelSha256 = kModelHash;
  std::string modelExportSha256 (64, '0');
  std::string actionGridHash = kGridHash;
  std::string patchSha256 (64, '0');
  std::string scenarioSha256 (64, '0');
  std::string outputDb = "dblbt-nru-wifi-validation.db";
  std::string nodeRate = "2Mbps";
  uint32_t wifiAps = 4;
  uint32_t nruGnbs = 4;
  uint32_t seed = 410;
  uint32_t runId = 1;
  uint32_t interferenceIntervalMs = 0;
  uint32_t interferenceDurationMs = 2;
  double simTime = 2.0;
  double appStart = 0.2;
  bool formal = false;

  CommandLine cmd;
  cmd.AddValue ("policy", "random, tmc, or adaptive", policy);
  cmd.AddValue ("scenario", "validation scenario id", scenario);
  cmd.AddValue ("wifiAps", "number of Wi-Fi AP/STA pairs", wifiAps);
  cmd.AddValue ("nruGnbs", "number of NR-U gNB/UE pairs", nruGnbs);
  cmd.AddValue ("simTime", "simulation duration in seconds", simTime);
  cmd.AddValue ("appStart", "measurement application start in seconds", appStart);
  cmd.AddValue ("seed", "ns-3 RNG seed", seed);
  cmd.AddValue ("runId", "ns-3 RNG run", runId);
  cmd.AddValue ("modelPath", "strict ASCII fixed LinUCB model", modelPath);
  cmd.AddValue ("modelSha256", "source NPZ SHA-256", modelSha256);
  cmd.AddValue ("modelExportSha256", "ASCII model SHA-256", modelExportSha256);
  cmd.AddValue ("actionGridHash", "24-arm grid SHA-256", actionGridHash);
  cmd.AddValue ("patchSha256", "applied source patch SHA-256", patchSha256);
  cmd.AddValue ("scenarioSha256", "scenario source SHA-256", scenarioSha256);
  cmd.AddValue ("outputDb", "output SQLite database", outputDb);
  cmd.AddValue ("nodeRate", "saturated offered load per contender", nodeRate);
  cmd.AddValue ("interferenceIntervalMs", "periodic interference interval", interferenceIntervalMs);
  cmd.AddValue ("interferenceDurationMs", "periodic interference duration", interferenceDurationMs);
  cmd.AddValue ("formal", "enforce the preregistered formal matrix", formal);
  cmd.Parse (argc, argv);

  NS_ABORT_MSG_IF (policy != "random" && policy != "tmc" && policy != "adaptive",
                   "unsupported policy");
  NS_ABORT_MSG_IF (wifiAps == 0 || nruGnbs == 0 || simTime <= appStart,
                   "invalid topology or timing");
  NS_ABORT_MSG_IF (!IsHash (modelSha256) || !IsHash (modelExportSha256) ||
                     !IsHash (actionGridHash) || !IsHash (patchSha256) ||
                     !IsHash (scenarioSha256),
                   "all provenance hashes must be lowercase SHA-256");
  NS_ABORT_MSG_IF (modelSha256 != kModelHash || actionGridHash != kGridHash,
                   "model provenance differs from the frozen experiment");
  if (policy == "adaptive")
    {
      NS_ABORT_MSG_IF (modelPath.empty (), "adaptive policy requires modelPath");
    }
  if (formal)
    {
      bool staticOk = scenario == "static-4x4" && wifiAps == 4 && nruGnbs == 4 &&
                      interferenceIntervalMs == 0;
      bool dynamicOk = scenario == "dynamic-4x4" && wifiAps == 4 && nruGnbs == 4 &&
                       interferenceIntervalMs == 0;
      bool nonidealOk = scenario == "nonideal-6x6-300ms" && wifiAps == 6 &&
                        nruGnbs == 6 && interferenceIntervalMs == 300 &&
                        interferenceDurationMs == 2;
      NS_ABORT_MSG_IF (!(staticOk || dynamicOk || nonidealOk),
                       "job is outside the frozen formal matrix");
      NS_ABORT_MSG_IF (seed != 410 && seed != 523 && seed != 631,
                       "formal seed is not preregistered");
      NS_ABORT_MSG_IF (runId != 1 || simTime != 2.0 || appStart != 0.2,
                       "formal timing or run id changed");
      NS_ABORT_MSG_IF (nodeRate != "2Mbps",
                       "formal offered load changed");
    }

  RngSeedManager::SetSeed (seed);
  RngSeedManager::SetRun (runId);
  ConfigureDefaults ();
  if (policy != "random")
    {
      Config::SetDefault ("ns3::NrDbLbtAccessManager::Policy", StringValue (policy));
      Config::SetDefault ("ns3::NrDbLbtAccessManager::ModelPath", StringValue (modelPath));
      Config::SetDefault ("ns3::NrDbLbtAccessManager::ModelSha256", StringValue (modelSha256));
      Config::SetDefault ("ns3::NrDbLbtAccessManager::ActionGridHash", StringValue (actionGridHash));
    }

  NodeContainer wifiApNodes;
  NodeContainer wifiStaNodes;
  NodeContainer nruGnbNodes;
  NodeContainer nruUeNodes;
  wifiApNodes.Create (wifiAps);
  wifiStaNodes.Create (wifiAps);
  nruGnbNodes.Create (nruGnbs);
  nruUeNodes.Create (nruGnbs);
  InstallPositions (wifiApNodes, wifiStaNodes, nruGnbNodes, nruUeNodes);

  Ptr<MultiModelSpectrumChannel> channel = CreateObject<MultiModelSpectrumChannel> ();
  Ptr<ThreeGppPropagationLossModel> propagation =
    CreateObject<ThreeGppIndoorOfficePropagationLossModel> ();
  Ptr<ThreeGppSpectrumPropagationLossModel> spectrumPropagation =
    CreateObject<ThreeGppSpectrumPropagationLossModel> ();
  propagation->SetAttributeFailSafe ("Frequency", DoubleValue (kFrequency));
  spectrumPropagation->SetChannelModelAttribute ("Frequency", DoubleValue (kFrequency));
  Ptr<ThreeGppIndoorMixedOfficeChannelConditionModel> condition =
    CreateObject<ThreeGppIndoorMixedOfficeChannelConditionModel> ();
  spectrumPropagation->SetChannelModelAttribute ("Scenario", StringValue ("InH-OfficeMixed"));
  spectrumPropagation->SetChannelModelAttribute ("ChannelConditionModel", PointerValue (condition));
  propagation->SetChannelConditionModel (condition);
  channel->AddPropagationLossModel (propagation);
  channel->AddSpectrumPropagationLossModel (spectrumPropagation);

  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  InternetStackHelper internet;
  internet.Install (remoteHostContainer);

  std::string semaphore = "/dblbt-" + scenario + "-" +
                          std::to_string (seed) + "-" + policy + "-base";
  SqliteOutputManager manager (outputDb, semaphore, 5.0, seed, runId);
  g_output = &manager;
  std::vector<ControllerBinding> bindings;
  std::set<uint32_t> wifiAddresses;
  std::set<uint32_t> nruAddresses;
  std::vector<Ipv4Address> destinations;
  std::vector<std::string> destinationTechnologies;
  std::vector<std::unique_ptr<WifiSetup>> wifiSetups;

  for (uint32_t index = 0; index < wifiAps; ++index)
    {
      std::ostringstream ssid;
      ssid << "dblbt-" << index;
      auto setup = std::make_unique<WifiSetup> (
        NodeContainer (wifiApNodes.Get (index)),
        NodeContainer (wifiStaNodes.Get (index)),
        channel, propagation, spectrumPropagation,
        kFrequency, kBandwidth, 4.0, 2.0, -62.0, -62.0,
        WIFI_STANDARD_80211ax_5GHZ, ssid.str ());
      std::unique_ptr<Ipv4AddressHelper> address = std::make_unique<Ipv4AddressHelper> ();
      std::ostringstream wifiNetwork;
      wifiNetwork << "10." << index << ".0.0";
      address->SetBase (wifiNetwork.str ().c_str (), "255.255.255.0");
      Ipv4InterfaceContainer staInterfaces = setup->AssignIpv4ToUe (address);
      setup->AssignIpv4ToStations (address);
      std::ostringstream backhaulNetwork;
      backhaulNetwork << "2." << index << ".0.0";
      setup->ConnectToRemotes (remoteHostContainer, backhaulNetwork.str ());
      Ipv4Address destination = staInterfaces.GetAddress (0);
      destinations.push_back (destination);
      destinationTechnologies.push_back ("wifi");
      wifiAddresses.insert (destination.Get ());
      setup->SetSinrCallback (MakeCallback (&OutputManager::SinrStore, &manager));
      setup->SetMacTxDataFailedCb (MakeCallback (&OutputManager::MacDataTxFailed, &manager));
      setup->SetChannelOccupancyCallback (
        MakeBoundCallback (&ObserveOccupancy, RadioKind::Wifi));

      Ptr<WifiNetDevice> device = DynamicCast<WifiNetDevice> (setup->GetGnbDev ().Get (0));
      uint32_t nodeId = device->GetNode ()->GetId ();
      if (policy == "random")
        {
          bindings.push_back ({nodeId, "wifi", "stock-wifi-" + std::to_string (nodeId), nullptr});
        }
      else
        {
          Ptr<DbLbtLocalController> controller = ConfigureController (
            policy, modelPath, modelSha256, actionGridHash, 1000 + nodeId * 2);
          BindWifiController (device, controller);
          bindings.push_back ({nodeId, "wifi", controller->GetStateId (), controller});
        }
      wifiSetups.push_back (std::move (setup));
    }

  std::unordered_map<uint32_t, uint32_t> connections;
  for (uint32_t index = 0; index < nruGnbs; ++index)
    {
      connections[index] = index;
    }
  std::string gnbCam = policy == "random"
                         ? "ns3::NrCat4LbtAccessManager"
                         : "ns3::NrDbLbtAccessManager";
  auto nrSetup = std::make_unique<NrSingleBwpSetup> (
    nruGnbNodes, nruUeNodes, channel, propagation, spectrumPropagation,
    kFrequency, kBandwidth, 0, 4.0, 2.0, connections,
    gnbCam, "ns3::NrAlwaysOnAccessManager",
    "ns3::NrMacSchedulerTdmaPF", BandwidthPartInfo::InH_OfficeMixed);
  std::unique_ptr<Ipv4AddressHelper> nrAddress = std::make_unique<Ipv4AddressHelper> ();
  Ipv4InterfaceContainer nrInterfaces = nrSetup->AssignIpv4ToUe (nrAddress);
  nrSetup->AssignIpv4ToStations (nrAddress);
  nrSetup->ConnectToRemotes (remoteHostContainer, "1.0.0.0");
  nrSetup->SetSinrCallback (MakeCallback (&OutputManager::SinrStore, &manager));
  nrSetup->SetMacTxDataFailedCb (MakeCallback (&OutputManager::MacDataTxFailed, &manager));
  nrSetup->SetChannelOccupancyCallback (
    MakeBoundCallback (&ObserveOccupancy, RadioKind::Nru));
  for (uint32_t index = 0; index < nruGnbs; ++index)
    {
      Ipv4Address destination = nrInterfaces.GetAddress (index);
      destinations.push_back (destination);
      destinationTechnologies.push_back ("nru");
      nruAddresses.insert (destination.Get ());
      Ptr<NrGnbNetDevice> device =
        DynamicCast<NrGnbNetDevice> (nrSetup->GetGnbDev ().Get (index));
      Ptr<NrGnbPhy> phy = device->GetPhy (0);
      uint32_t nodeId = device->GetNode ()->GetId ();
      if (policy == "random")
        {
          bindings.push_back ({nodeId, "nru", "stock-nru-" + std::to_string (nodeId), nullptr});
        }
      else
        {
          Ptr<NrDbLbtAccessManager> managerPtr =
            DynamicCast<NrDbLbtAccessManager> (phy->GetCam ());
          NS_ABORT_MSG_IF (!managerPtr, "gNB lacks NrDbLbtAccessManager");
          managerPtr->AssignStreams (2000 + nodeId * 2);
          phy->GetSpectrumPhy ()->TraceConnectWithoutContext (
            "TxDataTrace",
            MakeCallback (&NrDbLbtAccessManager::NotifyOwnTx, managerPtr));
          bindings.push_back (
            {nodeId, "nru", managerPtr->GetController ()->GetStateId (),
             managerPtr->GetController ()});
        }
    }

  Ipv4StaticRoutingHelper routingHelper;
  Ptr<Ipv4StaticRouting> remoteRouting =
    routingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteRouting->AddNetworkRouteTo (
    Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"),
    1 + wifiAps);
  for (uint32_t index = 0; index < wifiAps; ++index)
    {
      std::ostringstream network;
      network << "10." << index << ".0.0";
      remoteRouting->AddNetworkRouteTo (
        Ipv4Address (network.str ().c_str ()),
        Ipv4Mask ("255.255.255.0"), 1 + index);
    }

  uint16_t port = 1234;
  PacketSinkHelper sinkHelper (
    "ns3::UdpSocketFactory",
    Address (InetSocketAddress (Ipv4Address::GetAny (), port)));
  ApplicationContainer servers;
  servers.Add (sinkHelper.Install (wifiStaNodes));
  servers.Add (sinkHelper.Install (nruUeNodes));
  servers.Start (Seconds (0));
  servers.Stop (Seconds (simTime));

  ApplicationContainer clients;
  bool dynamic = scenario == "dynamic-4x4";
  double joinTime = appStart + (simTime - appStart) / 2.0;
  for (uint32_t index = 0; index < destinations.size (); ++index)
    {
      OnOffHelper onoff (
        "ns3::UdpSocketFactory",
        Address (InetSocketAddress (destinations.at (index), port)));
      onoff.SetConstantRate (DataRate (nodeRate), kPacketSize);
      ApplicationContainer installed = onoff.Install (remoteHost);
      uint32_t localIndex = destinationTechnologies.at (index) == "wifi"
                              ? index
                              : index - wifiAps;
      bool late = dynamic && localIndex >=
                              (destinationTechnologies.at (index) == "wifi"
                                 ? wifiAps / 2
                                 : nruGnbs / 2);
      installed.Start (Seconds (late ? joinTime : appStart));
      installed.Stop (Seconds (simTime));
      clients.Add (installed);
    }

  if (scenario == "nonideal-6x6-300ms" || interferenceIntervalMs > 0)
    {
      InstallInterferer (channel, spectrumPropagation, appStart, simTime,
                         interferenceIntervalMs, interferenceDurationMs);
    }

  FlowMonitorHelper flowHelper;
  Ptr<FlowMonitor> monitor = flowHelper.InstallAll ();
  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  monitor->CheckForLostPackets ();

  struct Aggregate
  {
    uint64_t txPackets {0};
    uint64_t rxPackets {0};
    uint64_t rxBytes {0};
    Time delay {Seconds (0)};
  };
  std::map<std::string, Aggregate> aggregates;
  aggregates["wifi"] = {};
  aggregates["nru"] = {};
  Ptr<Ipv4FlowClassifier> classifier =
    DynamicCast<Ipv4FlowClassifier> (flowHelper.GetClassifier ());
  for (const auto &entry : monitor->GetFlowStats ())
    {
      Ipv4FlowClassifier::FiveTuple tuple = classifier->FindFlow (entry.first);
      std::string technology;
      if (wifiAddresses.count (tuple.destinationAddress.Get ()) > 0)
        {
          technology = "wifi";
        }
      else if (nruAddresses.count (tuple.destinationAddress.Get ()) > 0)
        {
          technology = "nru";
        }
      else
        {
          continue;
        }
      const auto &stats = entry.second;
      Aggregate &aggregate = aggregates[technology];
      aggregate.txPackets += stats.txPackets;
      aggregate.rxPackets += stats.rxPackets;
      aggregate.rxBytes += stats.rxBytes;
      aggregate.delay += stats.delaySum;
      double throughput = stats.rxBytes * 8.0 / (simTime - appStart) / 1e6;
      double delay = stats.rxPackets > 0
                       ? stats.delaySum.GetMicroSeconds () /
                           static_cast<double> (stats.rxPackets)
                       : 0.0;
      double jitter = stats.rxPackets > 1
                        ? stats.jitterSum.GetMicroSeconds () /
                            static_cast<double> (stats.rxPackets - 1)
                        : 0.0;
      std::ostringstream address;
      tuple.destinationAddress.Print (address);
      manager.StoreE2EStatsFor (
        technology, throughput, stats.txBytes, stats.rxBytes,
        delay, jitter, address.str ());
    }

  double measurementDuration = simTime - appStart;
  TechnologyMetrics wifiMetrics;
  TechnologyMetrics nruMetrics;
  for (const auto &technology : {std::string ("wifi"), std::string ("nru")})
    {
      const Aggregate &aggregate = aggregates.at (technology);
      TechnologyMetrics value;
      value.throughputMbps = aggregate.rxBytes * 8.0 /
                             measurementDuration / 1e6;
      value.meanDelayUs = aggregate.rxPackets > 0
                            ? aggregate.delay.GetMicroSeconds () /
                                static_cast<double> (aggregate.rxPackets)
                            : 0.0;
      value.collisionProbability = aggregate.txPackets > 0
                                     ? static_cast<double> (
                                         aggregate.txPackets - aggregate.rxPackets) /
                                         aggregate.txPackets
                                     : 0.0;
      value.channelOccupancy = std::min (
        (technology == "wifi" ? g_wifiOccupancy.total : g_nruOccupancy.total)
          .GetSeconds () / measurementDuration,
        1.0);
      if (technology == "wifi")
        {
          wifiMetrics = value;
        }
      else
        {
          nruMetrics = value;
        }
    }
  manager.StoreChannelOccupancyRateFor ("wifi", wifiMetrics.channelOccupancy);
  manager.StoreChannelOccupancyRateFor ("nru", nruMetrics.channelOccupancy);
  manager.Close ();

  std::map<std::string, TechnologyMetrics> metrics {
    {"wifi", wifiMetrics}, {"nru", nruMetrics}};
  std::string jobId = scenario + "__seed-" + std::to_string (seed) + "__" + policy;
  WriteValidationOutput (
    outputDb, jobId, policy, scenario, seed, runId, wifiAps, nruGnbs,
    DataRate (nodeRate).GetBitRate (), modelSha256, modelExportSha256,
    actionGridHash, patchSha256,
    scenarioSha256, bindings, metrics);

  Simulator::Destroy ();
  return 0;
}
