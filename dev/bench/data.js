window.BENCHMARK_DATA = {
  "lastUpdate": 1772247478276,
  "repoUrl": "https://github.com/AreteDriver/animus",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "AreteDriver@users.noreply.github.com",
            "name": "AreteDriver",
            "username": "AreteDriver"
          },
          "committer": {
            "email": "AreteDriver@users.noreply.github.com",
            "name": "AreteDriver",
            "username": "AreteDriver"
          },
          "distinct": true,
          "id": "303635e27e0e77430022f4cba87d51b943b14f4b",
          "message": "fix: include workflow_dispatch in benchmarks job condition\n\nBenchmarks were skipped on manual workflow triggers. Include\nworkflow_dispatch so benchmarks can be tested without a code push.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-27T18:46:59-08:00",
          "tree_id": "ed56a245c9e432ff9b6d916456fb428c92cec9fb",
          "url": "https://github.com/AreteDriver/animus/commit/303635e27e0e77430022f4cba87d51b943b14f4b"
        },
        "date": 1772247477313,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 510.9759058644147,
            "unit": "iter/sec",
            "range": "stddev: 0.000025604817091575164",
            "extra": "mean: 1.9570394386958547 msec\nrounds: 522"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 638.3170761771171,
            "unit": "iter/sec",
            "range": "stddev: 0.00008721895690166745",
            "extra": "mean: 1.566619533334441 msec\nrounds: 645"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 208.15313885219336,
            "unit": "iter/sec",
            "range": "stddev: 0.000040663912231335155",
            "extra": "mean: 4.804155274881952 msec\nrounds: 211"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 288.8600679525777,
            "unit": "iter/sec",
            "range": "stddev: 0.00008697812844705218",
            "extra": "mean: 3.4618838356161103 msec\nrounds: 292"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1960.3949281477264,
            "unit": "iter/sec",
            "range": "stddev: 0.000013165702496177146",
            "extra": "mean: 510.1012993054655 usec\nrounds: 2018"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3332.2071592713746,
            "unit": "iter/sec",
            "range": "stddev: 0.00000639422222198918",
            "extra": "mean: 300.1013899203858 usec\nrounds: 3393"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 21635.089293697132,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025743710902219385",
            "extra": "mean: 46.221209740572974 usec\nrounds: 22914"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 346.0218597922688,
            "unit": "iter/sec",
            "range": "stddev: 0.00005167731551283308",
            "extra": "mean: 2.889990824858121 msec\nrounds: 354"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4551.07683996529,
            "unit": "iter/sec",
            "range": "stddev: 0.000006418801853360245",
            "extra": "mean: 219.72821711523264 usec\nrounds: 4721"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 35.05876114815298,
            "unit": "iter/sec",
            "range": "stddev: 0.00011482243595060682",
            "extra": "mean: 28.523540685711982 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 9.295426178856863,
            "unit": "iter/sec",
            "range": "stddev: 0.002713254987487083",
            "extra": "mean: 107.57979039998986 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1010.4118239309321,
            "unit": "iter/sec",
            "range": "stddev: 0.000060021648443086966",
            "extra": "mean: 989.695465072424 usec\nrounds: 1045"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 44.09648265655314,
            "unit": "iter/sec",
            "range": "stddev: 0.001969783872404429",
            "extra": "mean: 22.677545685186093 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 9361.758631612378,
            "unit": "iter/sec",
            "range": "stddev: 0.000004209565046023289",
            "extra": "mean: 106.81753710496696 usec\nrounds: 9756"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 702.4368287259019,
            "unit": "iter/sec",
            "range": "stddev: 0.00002421393713091917",
            "extra": "mean: 1.4236155610089891 msec\nrounds: 713"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 286.5617173137886,
            "unit": "iter/sec",
            "range": "stddev: 0.000043130372529968376",
            "extra": "mean: 3.489649662117943 msec\nrounds: 293"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 186396.93154177113,
            "unit": "iter/sec",
            "range": "stddev: 7.086837111943066e-7",
            "extra": "mean: 5.364895182171506 usec\nrounds: 194175"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 85286.68853072656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011053899460356196",
            "extra": "mean: 11.725159192219385 usec\nrounds: 89929"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 200.53932572082908,
            "unit": "iter/sec",
            "range": "stddev: 0.00005450321799607858",
            "extra": "mean: 4.986553118225303 msec\nrounds: 203"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 827.6550066332583,
            "unit": "iter/sec",
            "range": "stddev: 0.000021421746820838244",
            "extra": "mean: 1.2082328892901983 msec\nrounds: 831"
          }
        ]
      }
    ]
  }
}