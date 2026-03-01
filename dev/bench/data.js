window.BENCHMARK_DATA = {
  "lastUpdate": 1772364370596,
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
          "id": "4996bf4166e352d1fffb34c7ef77924db6fde659",
          "message": "fix: resolve remaining CodeQL alerts — empty-except, import patterns, unused vars\n\n- Add logger.debug to asyncio.CancelledError handlers in 6 channel adapters and proactive engine\n- Eliminate import-and-import-from patterns in test files (use sys.modules or top-level imports)\n- Reference _memory_manager in reflection check to clear unused-global-variable\n- Remove redundant local `import asyncio` in test_tools.py\n- Move TYPE_CHECKING imports to top-level in channels/base.py (no circular risk)\n- Fix chromadb_backend.py unused import by using module-level chromadb directly\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-27T22:08:45-08:00",
          "tree_id": "fe4162da07465e5a8c23aaf6ad0010c51c431af0",
          "url": "https://github.com/AreteDriver/animus/commit/4996bf4166e352d1fffb34c7ef77924db6fde659"
        },
        "date": 1772259580533,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 510.597878986077,
            "unit": "iter/sec",
            "range": "stddev: 0.0002076745343876603",
            "extra": "mean: 1.9584883548395393 msec\nrounds: 527"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 649.9781002817934,
            "unit": "iter/sec",
            "range": "stddev: 0.00003093489117752326",
            "extra": "mean: 1.5385133738605303 msec\nrounds: 658"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 209.2549618954748,
            "unit": "iter/sec",
            "range": "stddev: 0.00010988667405547207",
            "extra": "mean: 4.77885920095654 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 290.1140861579831,
            "unit": "iter/sec",
            "range": "stddev: 0.00006134089431920725",
            "extra": "mean: 3.4469198419253764 msec\nrounds: 291"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1952.855998552823,
            "unit": "iter/sec",
            "range": "stddev: 0.000014382474669888284",
            "extra": "mean: 512.0705268289402 usec\nrounds: 2050"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3303.558793949884,
            "unit": "iter/sec",
            "range": "stddev: 0.000006929415132423142",
            "extra": "mean: 302.7038604039357 usec\nrounds: 3417"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 21856.298050625202,
            "unit": "iter/sec",
            "range": "stddev: 0.000002710598858462347",
            "extra": "mean: 45.753402414431065 usec\nrounds: 22862"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 342.70277111756633,
            "unit": "iter/sec",
            "range": "stddev: 0.00005208743920197486",
            "extra": "mean: 2.9179804900291972 msec\nrounds: 351"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4483.310367103589,
            "unit": "iter/sec",
            "range": "stddev: 0.000007709051456924944",
            "extra": "mean: 223.04946972610398 usec\nrounds: 4707"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 35.29856267047245,
            "unit": "iter/sec",
            "range": "stddev: 0.0004354597732968551",
            "extra": "mean: 28.329765416666913 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 10.883518775705781,
            "unit": "iter/sec",
            "range": "stddev: 0.004760468684127189",
            "extra": "mean: 91.88204849999455 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1017.1003837946536,
            "unit": "iter/sec",
            "range": "stddev: 0.000023740164925407224",
            "extra": "mean: 983.187122857181 usec\nrounds: 1050"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 50.28152939371197,
            "unit": "iter/sec",
            "range": "stddev: 0.001430282441301626",
            "extra": "mean: 19.88801876271203 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 9076.734328892282,
            "unit": "iter/sec",
            "range": "stddev: 0.000004236457958577381",
            "extra": "mean: 110.17178246771923 usec\nrounds: 9240"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 677.8649254114305,
            "unit": "iter/sec",
            "range": "stddev: 0.00020195386131861696",
            "extra": "mean: 1.4752201545065182 msec\nrounds: 699"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 272.68990512686594,
            "unit": "iter/sec",
            "range": "stddev: 0.00021659593243293916",
            "extra": "mean: 3.66716912213806 msec\nrounds: 262"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 179769.70874035178,
            "unit": "iter/sec",
            "range": "stddev: 7.43725641764641e-7",
            "extra": "mean: 5.562672415764649 usec\nrounds: 187970"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 86254.73201232139,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012412023805649403",
            "extra": "mean: 11.593566830132302 usec\nrounds: 90992"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 197.70997493538906,
            "unit": "iter/sec",
            "range": "stddev: 0.00005331839044558053",
            "extra": "mean: 5.05791374626797 msec\nrounds: 201"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 813.2682611882335,
            "unit": "iter/sec",
            "range": "stddev: 0.00004158839121154135",
            "extra": "mean: 1.2296065735295514 msec\nrounds: 816"
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
          "id": "2a49da57e3f0206962078b09864a2bbc6a6ad0e4",
          "message": "feat(bootstrap): add IdentityProposalManager, enhance README, polish metadata\n\nPort selective improvements from the standalone animus-bootstrap repo:\n\n- Add typed Proposal dataclass + IdentityProposalManager wrapping\n  ImprovementStore with identity-focused methods (create/approve/reject/\n  list_pending/history). Wire into identity_tools.py and proposals router.\n- Enhance README with Identity Files, Self-Improvement, Philosophy, and\n  Roadmap sections from standalone. Expand CLI reference.\n- Add project.urls, keywords, and OS-specific classifiers to pyproject.toml.\n- Add 56 new tests: 37 for proposal manager lifecycle, 19 targeting\n  lowest-coverage modules (system, feedback router, web tools, SSE logs,\n  Linux service lifecycle).\n\n1459 tests passing, 92% coverage, lint clean.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-27T22:25:29-08:00",
          "tree_id": "68c8058f84e5956ef32e40f3530a885e2291c2b9",
          "url": "https://github.com/AreteDriver/animus/commit/2a49da57e3f0206962078b09864a2bbc6a6ad0e4"
        },
        "date": 1772260614460,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 498.91822947458314,
            "unit": "iter/sec",
            "range": "stddev: 0.00036735239500057114",
            "extra": "mean: 2.0043364642200228 msec\nrounds: 545"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 642.0772075962619,
            "unit": "iter/sec",
            "range": "stddev: 0.00004718812745679302",
            "extra": "mean: 1.5574450987657544 msec\nrounds: 648"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 207.45619199945384,
            "unit": "iter/sec",
            "range": "stddev: 0.00020189050540013402",
            "extra": "mean: 4.820294783019214 msec\nrounds: 212"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 290.6295876789962,
            "unit": "iter/sec",
            "range": "stddev: 0.00008969276388175239",
            "extra": "mean: 3.4408058999984266 msec\nrounds: 290"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1921.4346111142731,
            "unit": "iter/sec",
            "range": "stddev: 0.00001263861277522395",
            "extra": "mean: 520.4444607251468 usec\nrounds: 1986"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3315.1197561746694,
            "unit": "iter/sec",
            "range": "stddev: 0.000015479967101122626",
            "extra": "mean: 301.6482279825402 usec\nrounds: 3395"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 22563.25036627866,
            "unit": "iter/sec",
            "range": "stddev: 0.000002599968999941233",
            "extra": "mean: 44.31985568420253 usec\nrounds: 23116"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 350.91383933267593,
            "unit": "iter/sec",
            "range": "stddev: 0.00003244168074426977",
            "extra": "mean: 2.849702371105326 msec\nrounds: 353"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4744.525828240557,
            "unit": "iter/sec",
            "range": "stddev: 0.000009168945099001485",
            "extra": "mean: 210.7692182952741 usec\nrounds: 4810"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 35.75690841026407,
            "unit": "iter/sec",
            "range": "stddev: 0.00011574430926204561",
            "extra": "mean: 27.966623638886762 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 11.234373844762963,
            "unit": "iter/sec",
            "range": "stddev: 0.006092857662817564",
            "extra": "mean: 89.01252653846498 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1037.480321129673,
            "unit": "iter/sec",
            "range": "stddev: 0.00001760879476193073",
            "extra": "mean: 963.873704043984 usec\nrounds: 1088"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 50.00889559029051,
            "unit": "iter/sec",
            "range": "stddev: 0.0017001301717835286",
            "extra": "mean: 19.9964423968234 msec\nrounds: 63"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 9884.63992030409,
            "unit": "iter/sec",
            "range": "stddev: 0.000005207661858705543",
            "extra": "mean: 101.16706405722428 usec\nrounds: 10116"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 701.829223503936,
            "unit": "iter/sec",
            "range": "stddev: 0.00001806758685182765",
            "extra": "mean: 1.4248480492268814 msec\nrounds: 711"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 291.30020127856847,
            "unit": "iter/sec",
            "range": "stddev: 0.000080391161919114",
            "extra": "mean: 3.4328846860071565 msec\nrounds: 293"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 185607.6620337085,
            "unit": "iter/sec",
            "range": "stddev: 7.681232472700923e-7",
            "extra": "mean: 5.387708616352208 usec\nrounds: 189036"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 83984.15591740595,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011907142179570535",
            "extra": "mean: 11.90700780494178 usec\nrounds: 88021"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 202.18666062802953,
            "unit": "iter/sec",
            "range": "stddev: 0.00003606943298663523",
            "extra": "mean: 4.945924705882244 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 828.0723770557903,
            "unit": "iter/sec",
            "range": "stddev: 0.00001495918445600076",
            "extra": "mean: 1.2076239078949815 msec\nrounds: 836"
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
          "id": "9e7eecde5ae06651f49fc3305c719beb698cfafc",
          "message": "fix: resolve 3 CodeQL alerts in proposals — stack trace exposure, log injection\n\n- Replace exception message in HTMLResponse with generic \"Proposal not found\"\n- Remove user-controlled reason from log message (log injection)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-27T22:57:53-08:00",
          "tree_id": "3694f429470baebac90f4c52cdec862f20b1d2e0",
          "url": "https://github.com/AreteDriver/animus/commit/9e7eecde5ae06651f49fc3305c719beb698cfafc"
        },
        "date": 1772262541245,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 543.2340100829624,
            "unit": "iter/sec",
            "range": "stddev: 0.000015903247555869525",
            "extra": "mean: 1.8408273072727546 msec\nrounds: 550"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 712.1845449702553,
            "unit": "iter/sec",
            "range": "stddev: 0.000023438953436738456",
            "extra": "mean: 1.404130442119838 msec\nrounds: 717"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 225.83899764388372,
            "unit": "iter/sec",
            "range": "stddev: 0.000480991598120904",
            "extra": "mean: 4.427933219827955 msec\nrounds: 232"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 324.0903720943427,
            "unit": "iter/sec",
            "range": "stddev: 0.00005506919416582628",
            "extra": "mean: 3.085559109756275 msec\nrounds: 328"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 2147.6764176314864,
            "unit": "iter/sec",
            "range": "stddev: 0.000006299696443691835",
            "extra": "mean: 465.61949080896744 usec\nrounds: 2176"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3932.297789071675,
            "unit": "iter/sec",
            "range": "stddev: 0.000005345222491847288",
            "extra": "mean: 254.30423982108357 usec\nrounds: 4028"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 24126.346132282823,
            "unit": "iter/sec",
            "range": "stddev: 0.000003165164854819409",
            "extra": "mean: 41.44846445114731 usec\nrounds: 24614"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 392.84512029143343,
            "unit": "iter/sec",
            "range": "stddev: 0.00002708877756268342",
            "extra": "mean: 2.545532446115524 msec\nrounds: 399"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 5487.5569404779,
            "unit": "iter/sec",
            "range": "stddev: 0.000010190836273845126",
            "extra": "mean: 182.23045534592887 usec\nrounds: 5565"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 43.15182765491103,
            "unit": "iter/sec",
            "range": "stddev: 0.00008407886722997302",
            "extra": "mean: 23.17398947727285 msec\nrounds: 44"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 15.195574503942613,
            "unit": "iter/sec",
            "range": "stddev: 0.0027813262685245283",
            "extra": "mean: 65.80863393750214 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1157.5664746447828,
            "unit": "iter/sec",
            "range": "stddev: 0.00001887744860853704",
            "extra": "mean: 863.88127326067 usec\nrounds: 1193"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 68.7473369000436,
            "unit": "iter/sec",
            "range": "stddev: 0.0012772450897245288",
            "extra": "mean: 14.546017999998568 msec\nrounds: 79"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 10334.929489977998,
            "unit": "iter/sec",
            "range": "stddev: 0.000008301198221184166",
            "extra": "mean: 96.75924745976462 usec\nrounds: 10333"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 720.0304416008239,
            "unit": "iter/sec",
            "range": "stddev: 0.000009873787104431813",
            "extra": "mean: 1.388830169147748 msec\nrounds: 739"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 288.93306958414365,
            "unit": "iter/sec",
            "range": "stddev: 0.00006164014036057174",
            "extra": "mean: 3.4610091584161085 msec\nrounds: 303"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 209527.06982274156,
            "unit": "iter/sec",
            "range": "stddev: 3.8830297557505024e-7",
            "extra": "mean: 4.772653007776003 usec\nrounds: 109135"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 110912.99741861381,
            "unit": "iter/sec",
            "range": "stddev: 7.104418572173925e-7",
            "extra": "mean: 9.016075872746871 usec\nrounds: 114758"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 218.98341631854132,
            "unit": "iter/sec",
            "range": "stddev: 0.00002699394592187964",
            "extra": "mean: 4.566555846153041 msec\nrounds: 221"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 966.0463245781231,
            "unit": "iter/sec",
            "range": "stddev: 0.000017429597463007903",
            "extra": "mean: 1.0351470468424013 msec\nrounds: 982"
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
          "id": "d4bcd4e63c62f4ca4d839ed4dcbef47e39c55bb5",
          "message": "docs: update bootstrap test count and coverage in CLAUDE.md files\n\nBootstrap: 1459→1565 tests, 92→95% coverage, fail_under raised to 95.\nMonorepo total: 10,700+→10,900+.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <co-author>",
          "timestamp": "2026-02-28T00:17:22-08:00",
          "tree_id": "1768f62a2fc860729dbd5737797275b500a9ec68",
          "url": "https://github.com/AreteDriver/animus/commit/d4bcd4e63c62f4ca4d839ed4dcbef47e39c55bb5"
        },
        "date": 1772267314207,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 543.4872851515235,
            "unit": "iter/sec",
            "range": "stddev: 0.000014334444273091472",
            "extra": "mean: 1.8399694478983102 msec\nrounds: 547"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 688.5981100546983,
            "unit": "iter/sec",
            "range": "stddev: 0.00020596424312562333",
            "extra": "mean: 1.4522258853143901 msec\nrounds: 715"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 226.9397543034278,
            "unit": "iter/sec",
            "range": "stddev: 0.00009801047198816436",
            "extra": "mean: 4.40645581497792 msec\nrounds: 227"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 315.07135869695713,
            "unit": "iter/sec",
            "range": "stddev: 0.00013623576426023286",
            "extra": "mean: 3.173884177018524 msec\nrounds: 322"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 2147.6358054920943,
            "unit": "iter/sec",
            "range": "stddev: 0.00000811572272841447",
            "extra": "mean: 465.62829574862064 usec\nrounds: 2164"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3921.3330654389383,
            "unit": "iter/sec",
            "range": "stddev: 0.00001060728971598186",
            "extra": "mean: 255.0153183399799 usec\nrounds: 4024"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 24106.22710759072,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017212209497427797",
            "extra": "mean: 41.48305728378016 usec\nrounds: 24527"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 394.62762609982974,
            "unit": "iter/sec",
            "range": "stddev: 0.00003246260342980833",
            "extra": "mean: 2.534034451371704 msec\nrounds: 401"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 5474.153515101884,
            "unit": "iter/sec",
            "range": "stddev: 0.000003618476405587257",
            "extra": "mean: 182.67664530072798 usec\nrounds: 5554"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 41.29517677221897,
            "unit": "iter/sec",
            "range": "stddev: 0.00013722763392869333",
            "extra": "mean: 24.215903119047614 msec\nrounds: 42"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 16.418592452403136,
            "unit": "iter/sec",
            "range": "stddev: 0.0027980673516217846",
            "extra": "mean: 60.90656083333338 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1154.1984385044766,
            "unit": "iter/sec",
            "range": "stddev: 0.000020624111295298445",
            "extra": "mean: 866.4021425082889 usec\nrounds: 1228"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 74.72051452019805,
            "unit": "iter/sec",
            "range": "stddev: 0.0018818050822513388",
            "extra": "mean: 13.383205488095044 msec\nrounds: 84"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 10799.039193220744,
            "unit": "iter/sec",
            "range": "stddev: 0.000003393971549711135",
            "extra": "mean: 92.60083069499042 usec\nrounds: 10992"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 729.3107308591153,
            "unit": "iter/sec",
            "range": "stddev: 0.000016944477550953713",
            "extra": "mean: 1.371157666667015 msec\nrounds: 741"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 291.60231139104644,
            "unit": "iter/sec",
            "range": "stddev: 0.00008070702616383398",
            "extra": "mean: 3.42932809835987 msec\nrounds: 305"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 210340.7525785342,
            "unit": "iter/sec",
            "range": "stddev: 3.5455149820854225e-7",
            "extra": "mean: 4.754190463527192 usec\nrounds: 108992"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 111364.76204364697,
            "unit": "iter/sec",
            "range": "stddev: 6.640674346348094e-7",
            "extra": "mean: 8.97950107061758 usec\nrounds: 113020"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 219.86370530064116,
            "unit": "iter/sec",
            "range": "stddev: 0.00004377039068979658",
            "extra": "mean: 4.548272297297101 msec\nrounds: 222"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 963.7373920534982,
            "unit": "iter/sec",
            "range": "stddev: 0.000015029015238907803",
            "extra": "mean: 1.0376270633945568 msec\nrounds: 978"
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
          "id": "f2df237d9241aef5adaaebf1e6d0700682b559e2",
          "message": "test(bootstrap): push coverage from 95% to 96% with 41 targeted tests\n\nCover error handling paths in code_edit, timer_ctl, filesystem,\nidentity_tools, identity_page, home, web, self_improve, memory_tools,\nimprovement_store, and system. Raise fail_under gate from 80 to 96.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <co-author>",
          "timestamp": "2026-02-28T00:47:09-08:00",
          "tree_id": "c0caf595a7e823fe270cd2ab8175a7935219657c",
          "url": "https://github.com/AreteDriver/animus/commit/f2df237d9241aef5adaaebf1e6d0700682b559e2"
        },
        "date": 1772269162960,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 533.8496633069313,
            "unit": "iter/sec",
            "range": "stddev: 0.00001721217388423157",
            "extra": "mean: 1.8731865330878 msec\nrounds: 544"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 645.6061491409421,
            "unit": "iter/sec",
            "range": "stddev: 0.00006391174805432984",
            "extra": "mean: 1.5489319631335952 msec\nrounds: 651"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 209.32008782004752,
            "unit": "iter/sec",
            "range": "stddev: 0.00016116289619631438",
            "extra": "mean: 4.777372350711509 msec\nrounds: 211"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 290.0276616722775,
            "unit": "iter/sec",
            "range": "stddev: 0.000041994226048118623",
            "extra": "mean: 3.44794697938147 msec\nrounds: 291"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1971.8523997703037,
            "unit": "iter/sec",
            "range": "stddev: 0.000013545461211259335",
            "extra": "mean: 507.1373496903154 usec\nrounds: 2099"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3294.6573359718514,
            "unit": "iter/sec",
            "range": "stddev: 0.0000061243627802735496",
            "extra": "mean: 303.5217013562419 usec\nrounds: 3392"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 22375.755942958527,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028530289603442453",
            "extra": "mean: 44.691227529887854 usec\nrounds: 23153"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 351.6037385977809,
            "unit": "iter/sec",
            "range": "stddev: 0.00006079650454278272",
            "extra": "mean: 2.844110827683649 msec\nrounds: 354"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4658.499277783985,
            "unit": "iter/sec",
            "range": "stddev: 0.000005379063845579894",
            "extra": "mean: 214.66140496552632 usec\nrounds: 4793"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 35.30815468087918,
            "unit": "iter/sec",
            "range": "stddev: 0.00011917344495100721",
            "extra": "mean: 28.32206919444423 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 8.907387466141959,
            "unit": "iter/sec",
            "range": "stddev: 0.012874329914608683",
            "extra": "mean: 112.2663636000027 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1026.3250798792208,
            "unit": "iter/sec",
            "range": "stddev: 0.00001222662789856305",
            "extra": "mean: 974.3501543562409 usec\nrounds: 1056"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 38.75801236806436,
            "unit": "iter/sec",
            "range": "stddev: 0.004061843349927796",
            "extra": "mean: 25.80111669565324 msec\nrounds: 46"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 9272.781471456345,
            "unit": "iter/sec",
            "range": "stddev: 0.000003633507289949387",
            "extra": "mean: 107.84250691965721 usec\nrounds: 9538"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 708.345326164102,
            "unit": "iter/sec",
            "range": "stddev: 0.00004847924900972433",
            "extra": "mean: 1.4117408036208747 msec\nrounds: 718"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 288.961597919155,
            "unit": "iter/sec",
            "range": "stddev: 0.000049506500749550586",
            "extra": "mean: 3.46066746308545 msec\nrounds: 298"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 184646.69271114725,
            "unit": "iter/sec",
            "range": "stddev: 0.000002364847335697619",
            "extra": "mean: 5.4157482341931455 usec\nrounds: 189430"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 83408.26757346625,
            "unit": "iter/sec",
            "range": "stddev: 0.000001022171306466325",
            "extra": "mean: 11.989219163666204 usec\nrounds: 87405"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 202.58140024529504,
            "unit": "iter/sec",
            "range": "stddev: 0.000042650664960481026",
            "extra": "mean: 4.936287333334419 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 842.0503522997418,
            "unit": "iter/sec",
            "range": "stddev: 0.00002458393816398133",
            "extra": "mean: 1.1875774379392856 msec\nrounds: 854"
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
          "id": "fec2ad56a11c5380fe3c2348a8f36afe88be826b",
          "message": "test(forge): push coverage from 88% to 90% with 381 targeted tests\n\nAdd test_coverage_push_90.py covering 14+ modules: budget persistence,\nerrors, CLI commands (dev/coordination/admin/graph/helpers), field\nencryption, contracts enforcer, providers (anthropic/openai), websocket\nbroadcaster/manager, approval store, auto-parallel, executor AI/error,\nversioning, retry, rate limiters, agent context/memory, tracing,\nsafety, context window, scheduler, and version manager.\n\nFix bug in persistence.py where sqlite3 received None instead of empty\ntuple for params argument.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-28T22:14:11-08:00",
          "tree_id": "7cb1d34b25d2cc09fe228d4fed9851a4e35475b1",
          "url": "https://github.com/AreteDriver/animus/commit/fec2ad56a11c5380fe3c2348a8f36afe88be826b"
        },
        "date": 1772347785330,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 499.01171113022457,
            "unit": "iter/sec",
            "range": "stddev: 0.00021623468610858847",
            "extra": "mean: 2.0039609846732334 msec\nrounds: 522"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 650.7390105212038,
            "unit": "iter/sec",
            "range": "stddev: 0.000017695170329335074",
            "extra": "mean: 1.5367143875377298 msec\nrounds: 658"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 210.20826481541727,
            "unit": "iter/sec",
            "range": "stddev: 0.00047742317496775593",
            "extra": "mean: 4.7571868826284955 msec\nrounds: 213"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 289.0413215828692,
            "unit": "iter/sec",
            "range": "stddev: 0.000030279355145662894",
            "extra": "mean: 3.4597129383567955 msec\nrounds: 292"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1933.9289534124373,
            "unit": "iter/sec",
            "range": "stddev: 0.000011725831116972174",
            "extra": "mean: 517.0820769995142 usec\nrounds: 2013"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3348.9255713502116,
            "unit": "iter/sec",
            "range": "stddev: 0.000015757258094600676",
            "extra": "mean: 298.6032321992819 usec\nrounds: 3441"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 21831.95339652754,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035278701872461733",
            "extra": "mean: 45.8044217041547 usec\nrounds: 22862"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 350.59454453131906,
            "unit": "iter/sec",
            "range": "stddev: 0.00004517295627238861",
            "extra": "mean: 2.85229766292233 msec\nrounds: 356"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4774.746368452028,
            "unit": "iter/sec",
            "range": "stddev: 0.000010169611080917942",
            "extra": "mean: 209.43520824629684 usec\nrounds: 4778"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 34.178695349880016,
            "unit": "iter/sec",
            "range": "stddev: 0.000327144897217642",
            "extra": "mean: 29.25799214286014 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 11.407703642232663,
            "unit": "iter/sec",
            "range": "stddev: 0.0030019293754868213",
            "extra": "mean: 87.66006125000321 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1011.88058717736,
            "unit": "iter/sec",
            "range": "stddev: 0.00001818669033851013",
            "extra": "mean: 988.2589039379628 usec\nrounds: 1041"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 50.95189168779708,
            "unit": "iter/sec",
            "range": "stddev: 0.001457368868481635",
            "extra": "mean: 19.626356684211174 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 8775.103669564596,
            "unit": "iter/sec",
            "range": "stddev: 0.000004060865410708571",
            "extra": "mean: 113.95876762896616 usec\nrounds: 9317"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 700.4551097566331,
            "unit": "iter/sec",
            "range": "stddev: 0.00003294232218695424",
            "extra": "mean: 1.4276432366200327 msec\nrounds: 710"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 292.98645527395917,
            "unit": "iter/sec",
            "range": "stddev: 0.00003068525139688644",
            "extra": "mean: 3.4131270644062455 msec\nrounds: 295"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 186671.7369754498,
            "unit": "iter/sec",
            "range": "stddev: 7.915406687350206e-7",
            "extra": "mean: 5.356997348406928 usec\nrounds: 191205"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 85017.37918628762,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012842607862130383",
            "extra": "mean: 11.76230095036015 usec\nrounds: 90490"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 201.55023051395386,
            "unit": "iter/sec",
            "range": "stddev: 0.00003968025057155566",
            "extra": "mean: 4.961542328430963 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 828.6733737355485,
            "unit": "iter/sec",
            "range": "stddev: 0.00001639966027093308",
            "extra": "mean: 1.2067480767387688 msec\nrounds: 834"
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
          "id": "813adc12a71ff1d7b3fdb347c499baae7bb81f50",
          "message": "docs(forge): update coverage stats to 90% and raise gate\n\n- CLAUDE.md: 6731/85% → 7348/90%\n- pyproject.toml: fail_under 85 → 90\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-28T23:43:15-08:00",
          "tree_id": "46244eeaea53a2a39b45af5602c3b750f6feec2d",
          "url": "https://github.com/AreteDriver/animus/commit/813adc12a71ff1d7b3fdb347c499baae7bb81f50"
        },
        "date": 1772351656073,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 502.3346869012069,
            "unit": "iter/sec",
            "range": "stddev: 0.00022079000085357113",
            "extra": "mean: 1.990704655831716 msec\nrounds: 523"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 655.9648309067259,
            "unit": "iter/sec",
            "range": "stddev: 0.00002148158719585119",
            "extra": "mean: 1.5244719730137388 msec\nrounds: 667"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 207.30053158504583,
            "unit": "iter/sec",
            "range": "stddev: 0.00016610217531220605",
            "extra": "mean: 4.823914306219452 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 288.60139112537047,
            "unit": "iter/sec",
            "range": "stddev: 0.000032998024749029714",
            "extra": "mean: 3.4649867628863684 msec\nrounds: 291"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1880.8670789197236,
            "unit": "iter/sec",
            "range": "stddev: 0.000008687137045684668",
            "extra": "mean: 531.6696810783409 usec\nrounds: 1966"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3295.192276078236,
            "unit": "iter/sec",
            "range": "stddev: 0.000018524788453351044",
            "extra": "mean: 303.472427772909 usec\nrounds: 3399"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 22300.876709237473,
            "unit": "iter/sec",
            "range": "stddev: 0.000003127099384247424",
            "extra": "mean: 44.84128642286874 usec\nrounds: 23186"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 344.61821131206455,
            "unit": "iter/sec",
            "range": "stddev: 0.0001775931963333164",
            "extra": "mean: 2.901761912676353 msec\nrounds: 355"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4795.153254581206,
            "unit": "iter/sec",
            "range": "stddev: 0.000006216360968313378",
            "extra": "mean: 208.54390817324085 usec\nrounds: 5053"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 35.334461183814106,
            "unit": "iter/sec",
            "range": "stddev: 0.00020488784114008334",
            "extra": "mean: 28.300983416667375 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 11.575420062758289,
            "unit": "iter/sec",
            "range": "stddev: 0.004627985890850593",
            "extra": "mean: 86.38995341666345 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1019.2808192692178,
            "unit": "iter/sec",
            "range": "stddev: 0.00001562632611472973",
            "extra": "mean: 981.0838986619593 usec\nrounds: 1046"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 49.8395128526499,
            "unit": "iter/sec",
            "range": "stddev: 0.0013857385919766895",
            "extra": "mean: 20.064401571429716 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 9293.353893903435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034368990425368793",
            "extra": "mean: 107.6037791540483 usec\nrounds: 9124"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 701.4576613425643,
            "unit": "iter/sec",
            "range": "stddev: 0.000034205578299523694",
            "extra": "mean: 1.425602791316067 msec\nrounds: 714"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 287.8304811456066,
            "unit": "iter/sec",
            "range": "stddev: 0.00003983414693838389",
            "extra": "mean: 3.4742672006795687 msec\nrounds: 294"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 182416.55930198648,
            "unit": "iter/sec",
            "range": "stddev: 6.842298556876817e-7",
            "extra": "mean: 5.481958457206304 usec\nrounds: 188360"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 86420.31821222525,
            "unit": "iter/sec",
            "range": "stddev: 9.75262024030414e-7",
            "extra": "mean: 11.571352902731356 usec\nrounds: 90498"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 200.7341925276918,
            "unit": "iter/sec",
            "range": "stddev: 0.000037453963413110657",
            "extra": "mean: 4.981712320197007 msec\nrounds: 203"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 843.6306442009039,
            "unit": "iter/sec",
            "range": "stddev: 0.000021509247479612622",
            "extra": "mean: 1.1853528636897854 msec\nrounds: 851"
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
          "id": "c72e727268b023bcc749aeddbfe1e46ac650bc1c",
          "message": "docs(forge): add test count and coverage to CLAUDE.md\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-01T00:27:09-08:00",
          "tree_id": "cc9e4c05c5bcaad772a9edf5fa0f2170f18df002",
          "url": "https://github.com/AreteDriver/animus/commit/c72e727268b023bcc749aeddbfe1e46ac650bc1c"
        },
        "date": 1772354294199,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 533.8662942425216,
            "unit": "iter/sec",
            "range": "stddev: 0.00002199591444724512",
            "extra": "mean: 1.8731281798167352 msec\nrounds: 545"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 653.7442029207517,
            "unit": "iter/sec",
            "range": "stddev: 0.00002290941640851304",
            "extra": "mean: 1.5296502753405252 msec\nrounds: 661"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 209.30983979331953,
            "unit": "iter/sec",
            "range": "stddev: 0.00016616155601414007",
            "extra": "mean: 4.7776062558140495 msec\nrounds: 215"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 289.3589778161665,
            "unit": "iter/sec",
            "range": "stddev: 0.000024713839516221564",
            "extra": "mean: 3.455914890034319 msec\nrounds: 291"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1965.9622295687348,
            "unit": "iter/sec",
            "range": "stddev: 0.000028764236117324932",
            "extra": "mean: 508.6567712032626 usec\nrounds: 2028"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3332.084101364166,
            "unit": "iter/sec",
            "range": "stddev: 0.000008611009186495466",
            "extra": "mean: 300.11247302869594 usec\nrounds: 3374"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 22430.955921894525,
            "unit": "iter/sec",
            "range": "stddev: 0.000002434015423161067",
            "extra": "mean: 44.58124760630084 usec\nrounds: 23186"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 346.0784168068891,
            "unit": "iter/sec",
            "range": "stddev: 0.00004735373905927194",
            "extra": "mean: 2.8895185352110464 msec\nrounds: 355"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4846.235879835197,
            "unit": "iter/sec",
            "range": "stddev: 0.000006622606871193225",
            "extra": "mean: 206.34571341459477 usec\nrounds: 5084"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 35.65416734098382,
            "unit": "iter/sec",
            "range": "stddev: 0.00012211604011328484",
            "extra": "mean: 28.0472122777782 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 11.915144416445267,
            "unit": "iter/sec",
            "range": "stddev: 0.0038452124809286886",
            "extra": "mean: 83.92680483333474 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1027.344109280328,
            "unit": "iter/sec",
            "range": "stddev: 0.00008731014059839283",
            "extra": "mean: 973.3836900087128 usec\nrounds: 1071"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 51.24826956016117,
            "unit": "iter/sec",
            "range": "stddev: 0.0013953091384393131",
            "extra": "mean: 19.51285396721706 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 9136.430362799101,
            "unit": "iter/sec",
            "range": "stddev: 0.000003959561053743597",
            "extra": "mean: 109.4519369481226 usec\nrounds: 9516"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 701.3453457678717,
            "unit": "iter/sec",
            "range": "stddev: 0.000016795439432850402",
            "extra": "mean: 1.425831091678729 msec\nrounds: 709"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 292.7169004171699,
            "unit": "iter/sec",
            "range": "stddev: 0.00003107865460111357",
            "extra": "mean: 3.416270118243378 msec\nrounds: 296"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 184639.42949985614,
            "unit": "iter/sec",
            "range": "stddev: 7.883492632572501e-7",
            "extra": "mean: 5.4159612749495585 usec\nrounds: 188715"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 81993.30129295554,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011676516136876937",
            "extra": "mean: 12.196118271016793 usec\nrounds: 87866"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 202.2587313878435,
            "unit": "iter/sec",
            "range": "stddev: 0.00004773275561677545",
            "extra": "mean: 4.944162326828989 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 836.3859879440662,
            "unit": "iter/sec",
            "range": "stddev: 0.000023901488813966894",
            "extra": "mean: 1.1956202213025067 msec\nrounds: 845"
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
          "id": "aaf103bb98d966ca1af679786481336a9f6ac87c",
          "message": "test(core): push coverage from 95% to 97% with 88 targeted tests\n\nCover ChromaMemoryStore (mock chromadb), MemoryLayer edge cases\n(export/import/backup/consolidate/csv), cognitive provider error\npaths (Ollama/Anthropic/OpenAI), CognitiveLayer enrichment, and\nbootstrap loop convergent consensus. Raise fail_under gate to 97.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-01T01:02:03-08:00",
          "tree_id": "215fe88fa811ccb13c5207e945b3d5d79552ca72",
          "url": "https://github.com/AreteDriver/animus/commit/aaf103bb98d966ca1af679786481336a9f6ac87c"
        },
        "date": 1772356424017,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 525.7689115558617,
            "unit": "iter/sec",
            "range": "stddev: 0.00005196863257323862",
            "extra": "mean: 1.901976282775617 msec\nrounds: 534"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 694.1707256764372,
            "unit": "iter/sec",
            "range": "stddev: 0.00013503885300507868",
            "extra": "mean: 1.440567807041339 msec\nrounds: 710"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 227.29880357149256,
            "unit": "iter/sec",
            "range": "stddev: 0.00004439525968908379",
            "extra": "mean: 4.39949522077211 msec\nrounds: 231"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 325.6466368527676,
            "unit": "iter/sec",
            "range": "stddev: 0.00004716429734241063",
            "extra": "mean: 3.0708132276892615 msec\nrounds: 325"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1926.1426423797545,
            "unit": "iter/sec",
            "range": "stddev: 0.00000841421131345165",
            "extra": "mean: 519.1723489203777 usec\nrounds: 1946"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3713.3973085679286,
            "unit": "iter/sec",
            "range": "stddev: 0.000007623255301206542",
            "extra": "mean: 269.2951809095941 usec\nrounds: 3803"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 21920.05886393761,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033110960887512845",
            "extra": "mean: 45.62031544747252 usec\nrounds: 22923"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 391.531424917277,
            "unit": "iter/sec",
            "range": "stddev: 0.000027306320391960073",
            "extra": "mean: 2.55407340601404 msec\nrounds: 399"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4539.3660662859365,
            "unit": "iter/sec",
            "range": "stddev: 0.00000940674399185785",
            "extra": "mean: 220.29507763805222 usec\nrounds: 4727"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 36.16596422180774,
            "unit": "iter/sec",
            "range": "stddev: 0.00007950434709106913",
            "extra": "mean: 27.650306621633202 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 13.635948012669123,
            "unit": "iter/sec",
            "range": "stddev: 0.0016757691989493513",
            "extra": "mean: 73.33556853332841 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 984.5332469263608,
            "unit": "iter/sec",
            "range": "stddev: 0.000016979150142604697",
            "extra": "mean: 1.015709731613356 msec\nrounds: 1006"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 62.064878763644515,
            "unit": "iter/sec",
            "range": "stddev: 0.0007655928645354772",
            "extra": "mean: 16.112171971014398 msec\nrounds: 69"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 10540.440969710402,
            "unit": "iter/sec",
            "range": "stddev: 0.000004007860816445787",
            "extra": "mean: 94.87269108319622 usec\nrounds: 11058"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 721.7646847475033,
            "unit": "iter/sec",
            "range": "stddev: 0.00006184201986954046",
            "extra": "mean: 1.3854931131049066 msec\nrounds: 725"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 294.931003389826,
            "unit": "iter/sec",
            "range": "stddev: 0.000038700473723371015",
            "extra": "mean: 3.3906235306101293 msec\nrounds: 294"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 178090.79071251376,
            "unit": "iter/sec",
            "range": "stddev: 8.116525428761591e-7",
            "extra": "mean: 5.6151134822814495 usec\nrounds: 183218"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 87522.30325000566,
            "unit": "iter/sec",
            "range": "stddev: 0.000001220239804424604",
            "extra": "mean: 11.425659093357273 usec\nrounds: 90934"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 215.08557608333842,
            "unit": "iter/sec",
            "range": "stddev: 0.000039016661654978013",
            "extra": "mean: 4.649312232878572 msec\nrounds: 219"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 868.8034140235303,
            "unit": "iter/sec",
            "range": "stddev: 0.000010971910082284799",
            "extra": "mean: 1.1510083683590548 msec\nrounds: 866"
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
          "id": "4dfc9a6cebf5a6372e9a8c17fcca2d6757625c54",
          "message": "test(forge): add Ollama integration tests for self-improve pipeline\n\n7 test classes (11 methods) exercising the full pipeline with live\nOllama: provider connectivity, analyzer, code generation, sandbox\nvalidation, end-to-end orchestrator run, error recovery, and model\nperformance benchmarks. Excluded from CI via conftest.py collect_ignore.\nRun manually: pytest tests/test_self_improve_ollama_integration.py -v -s\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-01T02:19:57-08:00",
          "tree_id": "4febf167af3fb1aa18e1899dace9a08d5103846c",
          "url": "https://github.com/AreteDriver/animus/commit/4dfc9a6cebf5a6372e9a8c17fcca2d6757625c54"
        },
        "date": 1772364369997,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::TestMemorySerdeBenchmark::test_memory_serde_500",
            "value": 524.727970569564,
            "unit": "iter/sec",
            "range": "stddev: 0.00009217240339052534",
            "extra": "mean: 1.905749371268609 msec\nrounds: 536"
          },
          {
            "name": "tests/test_benchmarks.py::TestConversationSerdeBenchmark::test_conversation_serde_50msg",
            "value": 651.4348917006181,
            "unit": "iter/sec",
            "range": "stddev: 0.000044635567409138204",
            "extra": "mean: 1.5350728257576556 msec\nrounds: 660"
          },
          {
            "name": "tests/test_benchmarks.py::TestDetectModeBenchmark::test_detect_mode_1000",
            "value": 210.06018266248185,
            "unit": "iter/sec",
            "range": "stddev: 0.00015064287063127805",
            "extra": "mean: 4.760540466666017 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::TestDelegationDetectionBenchmark::test_delegation_detection_500",
            "value": 290.0451922314671,
            "unit": "iter/sec",
            "range": "stddev: 0.00013445086351364897",
            "extra": "mean: 3.447738582758379 msec\nrounds: 290"
          },
          {
            "name": "tests/test_benchmarks.py::TestEntitySearchBenchmark::test_entity_search_200",
            "value": 1955.9224612436963,
            "unit": "iter/sec",
            "range": "stddev: 0.000012841469189355887",
            "extra": "mean: 511.2677111771283 usec\nrounds: 2022"
          },
          {
            "name": "tests/test_benchmarks.py::TestLocalSearchBenchmark::test_local_search_500",
            "value": 3401.608153568594,
            "unit": "iter/sec",
            "range": "stddev: 0.00002594956623895155",
            "extra": "mean: 293.97859919606253 usec\nrounds: 3483"
          },
          {
            "name": "tests/test_benchmarks.py::TestWorkflowParseBenchmark::test_parse_20_step_workflow",
            "value": 22323.333603935098,
            "unit": "iter/sec",
            "range": "stddev: 0.000002782290120387769",
            "extra": "mean: 44.79617684984659 usec\nrounds: 22799"
          },
          {
            "name": "tests/test_benchmarks.py::TestYAMLLoadBenchmark::test_load_yaml_10_steps",
            "value": 337.31475107454264,
            "unit": "iter/sec",
            "range": "stddev: 0.00027531342108494644",
            "extra": "mean: 2.9645901841363926 msec\nrounds: 353"
          },
          {
            "name": "tests/test_benchmarks.py::TestConditionEvalBenchmark::test_condition_evaluate_1000",
            "value": 4541.117210964308,
            "unit": "iter/sec",
            "range": "stddev: 0.000006523673225739483",
            "extra": "mean: 220.2101274958392 usec\nrounds: 4808"
          },
          {
            "name": "tests/test_benchmarks.py::TestCacheBenchmark::test_cache_set_get_1000",
            "value": 35.35472750422922,
            "unit": "iter/sec",
            "range": "stddev: 0.0001116033209317828",
            "extra": "mean: 28.28476050000322 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::TestTaskStoreBenchmark::test_record_query_100",
            "value": 11.292446194971594,
            "unit": "iter/sec",
            "range": "stddev: 0.002377353425837806",
            "extra": "mean: 88.55477216666212 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::TestBudgetBenchmark::test_budget_record_usage_500",
            "value": 1041.1748550707161,
            "unit": "iter/sec",
            "range": "stddev: 0.000018774044455994347",
            "extra": "mean: 960.4534676666585 usec\nrounds: 1067"
          },
          {
            "name": "tests/test_benchmarks.py::TestCheckpointBenchmark::test_checkpoint_lifecycle_20",
            "value": 53.209198948202776,
            "unit": "iter/sec",
            "range": "stddev: 0.0012900749286716923",
            "extra": "mean: 18.793742807018457 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::TestSkillLookupBenchmark::test_skill_lookup_100",
            "value": 9183.002292558605,
            "unit": "iter/sec",
            "range": "stddev: 0.000004233201042331029",
            "extra": "mean: 108.89684747333064 usec\nrounds: 9795"
          },
          {
            "name": "tests/test_benchmarks.py::TestResolverBenchmark::test_resolve_50_intents",
            "value": 700.1398526887391,
            "unit": "iter/sec",
            "range": "stddev: 0.000029189275150587483",
            "extra": "mean: 1.4282860719322166 msec\nrounds: 709"
          },
          {
            "name": "tests/test_benchmarks.py::TestStructuralOverlapBenchmark::test_structural_overlaps_1000",
            "value": 296.119831404381,
            "unit": "iter/sec",
            "range": "stddev: 0.00006591214369071448",
            "extra": "mean: 3.377011243243621 msec\nrounds: 296"
          },
          {
            "name": "tests/test_benchmarks.py::TestConstraintBenchmark::test_validate_20_constraints",
            "value": 185839.55452254272,
            "unit": "iter/sec",
            "range": "stddev: 7.211511392910152e-7",
            "extra": "mean: 5.380985778669083 usec\nrounds: 190840"
          },
          {
            "name": "tests/test_benchmarks.py::TestPhiScoringBenchmark::test_phi_score_100_outcomes",
            "value": 87431.89966528707,
            "unit": "iter/sec",
            "range": "stddev: 9.975220499360316e-7",
            "extra": "mean: 11.437473094239863 usec\nrounds: 90910"
          },
          {
            "name": "tests/test_benchmarks.py::TestRealisticScenarioBenchmark::test_realistic_25_agents",
            "value": 203.08084545830008,
            "unit": "iter/sec",
            "range": "stddev: 0.00004627189783466759",
            "extra": "mean: 4.924147315534673 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::TestPublishThroughputBenchmark::test_publish_100_intents",
            "value": 838.7411721338912,
            "unit": "iter/sec",
            "range": "stddev: 0.000026891399276044763",
            "extra": "mean: 1.1922629211772693 msec\nrounds: 850"
          }
        ]
      }
    ]
  }
}