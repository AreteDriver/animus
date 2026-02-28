window.BENCHMARK_DATA = {
  "lastUpdate": 1772258713697,
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
      },
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
          "id": "6898d2de93999963cfe052b4712750970fc742bb",
          "message": "fix: resolve CodeQL alerts — XSS sanitizer, side-effects, empty-except\n\n- Switch html.escape to markupsafe.escape in identity_page.py (CodeQL\n  recognizes markupsafe as XSS sanitizer)\n- Extract side-effect calls from assert statements in test_personas.py,\n  test_self_mod_tools.py (py/side-effect-in-assert)\n- Remove unused result assignment in test_tools.py (py/multiple-definition)\n- Replace empty except:pass with explicit fallback values/logging in\n  identity.py, channels.py, forge_ctl.py, home.py, chat.py\n- Add all re-exported names to __all__ in forge CLI main.py\n  (py/unused-import)\n- Use task.result() instead of bare await in approval persistence tests\n  (py/ineffectual-statement)\n- Fix undefined variable pkg→pkg_name in scripts/chat.py\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-27T21:54:17-08:00",
          "tree_id": "6349963dafebb0e72bd7da96e2fea6f8a428284b",
          "url": "https://github.com/AreteDriver/animus/commit/6898d2de93999963cfe052b4712750970fc742bb"
        },
        "date": 1772258712861,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 532.1645055117494,
            "unit": "iter/sec",
            "range": "stddev: 0.000032466624409432265",
            "extra": "mean: 1.8791181855286692 msec\nrounds: 539"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 662.1361731756829,
            "unit": "iter/sec",
            "range": "stddev: 0.00006862280715737031",
            "extra": "mean: 1.5102633574660067 msec\nrounds: 663"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 206.50731535633992,
            "unit": "iter/sec",
            "range": "stddev: 0.0002077587335890381",
            "extra": "mean: 4.842443466346188 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 289.171203808279,
            "unit": "iter/sec",
            "range": "stddev: 0.00010708241209828882",
            "extra": "mean: 3.4581589965749204 msec\nrounds: 292"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1911.6727486896168,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135141647559048",
            "extra": "mean: 523.1020846457451 usec\nrounds: 2032"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3318.3925069002457,
            "unit": "iter/sec",
            "range": "stddev: 0.000008305078720247051",
            "extra": "mean: 301.3507286798068 usec\nrounds: 3424"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 22516.540680526687,
            "unit": "iter/sec",
            "range": "stddev: 0.000002486180576261464",
            "extra": "mean: 44.41179549684756 usec\nrounds: 23095"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 345.89295349417006,
            "unit": "iter/sec",
            "range": "stddev: 0.00003402971167718957",
            "extra": "mean: 2.8910678575499076 msec\nrounds: 351"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4811.042854071239,
            "unit": "iter/sec",
            "range": "stddev: 0.000006615513633951098",
            "extra": "mean: 207.85514291434174 usec\nrounds: 4982"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 34.22540783191376,
            "unit": "iter/sec",
            "range": "stddev: 0.0004921825016101205",
            "extra": "mean: 29.21805942857288 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 11.504741660580157,
            "unit": "iter/sec",
            "range": "stddev: 0.0031225263293241977",
            "extra": "mean: 86.92068275000035 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 996.9008709130336,
            "unit": "iter/sec",
            "range": "stddev: 0.00001841703186607531",
            "extra": "mean: 1.003108763546498 msec\nrounds: 1015"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 51.59839881369442,
            "unit": "iter/sec",
            "range": "stddev: 0.0010580882107868925",
            "extra": "mean: 19.38044635087777 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 9504.5766388828,
            "unit": "iter/sec",
            "range": "stddev: 0.0000042738386457258276",
            "extra": "mean: 105.2124716327758 usec\nrounds: 9800"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 690.8641799230907,
            "unit": "iter/sec",
            "range": "stddev: 0.000026238562106766083",
            "extra": "mean: 1.447462510086025 msec\nrounds: 694"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 291.20082740476994,
            "unit": "iter/sec",
            "range": "stddev: 0.00009491750697573745",
            "extra": "mean: 3.4340561766673736 msec\nrounds: 300"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 186681.80859753687,
            "unit": "iter/sec",
            "range": "stddev: 7.387721094222772e-7",
            "extra": "mean: 5.356708334425223 usec\nrounds: 191939"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 83148.67890990399,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011717057737341917",
            "extra": "mean: 12.026649287880486 usec\nrounds: 87559"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 200.17926568560765,
            "unit": "iter/sec",
            "range": "stddev: 0.000052469972950115875",
            "extra": "mean: 4.995522371285716 msec\nrounds: 202"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 822.0152243857272,
            "unit": "iter/sec",
            "range": "stddev: 0.000018998668524202247",
            "extra": "mean: 1.216522480769473 msec\nrounds: 832"
          }
        ]
      }
    ]
  }
}