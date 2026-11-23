#!/usr/bin/env python3
"""Known-answer tests only; no GPU, SSH, timing measurements, or file writes."""
import unittest
from analyze_engram_events import classify,analyze

def record(**kw):
 r=dict(request_id='fixture',rank=0,chunk=2048,layer=1,generation=7,data_generation=7,
 clock_domain='rank0.monotonic',clock_alignment_verified=True,t_need_ns=100_000_000,t_join_enter_ns=90_000_000,
 t_join_exit_ns=150_000_000,t_current_ready_ns=80_000_000,t_all_done_ns=150_000_000,
 wait_binding='whole_reader')
 r.update(kw);return r

class Cases(unittest.TestCase):
 def test_coupling(self):
  x=classify(record());self.assertTrue(x['coupling_pattern_supported']);self.assertEqual(x['hypothetical_coupling_without_new_cost_ms'],50)
 def test_current_supply(self):
  x=classify(record(t_current_ready_ns=130_000_000));self.assertFalse(x['coupling_pattern_supported']);self.assertEqual(x['current_supply_late_ms'],30);self.assertEqual(x['hypothetical_coupling_without_new_cost_ms'],20)
 def test_host_wait_hidden_by_gpu(self):
  x=classify(record(t_need_ns=200_000_000));self.assertEqual(x['join_host_ms'],60);self.assertEqual(x['release_gap_envelope_ms'],0);self.assertEqual(x['hypothetical_coupling_without_new_cost_ms'],0)
 def test_current_layer_binding(self):
  x=classify(record(wait_binding='current_layer'));self.assertFalse(x['coupling_pattern_supported']);self.assertEqual(x['mechanism'],'data_ready_but_release_late_other_cause')
 def test_missing(self):
  r=record();del r['t_need_ns'];self.assertEqual(classify(r)['status'],'unverified')
 def test_stale(self):self.assertEqual(classify(record(data_generation=6))['status'],'invalid')
 def test_bad_chronology(self):self.assertEqual(classify(record(t_all_done_ns=60_000_000))['status'],'invalid')
 def test_whole_join_early(self):self.assertEqual(classify(record(t_all_done_ns=160_000_000))['status'],'invalid')
 def test_clock(self):self.assertEqual(classify(record(clock_alignment_verified=False))['status'],'unverified')
 def test_uncertainty(self):
  x=classify(record(need_uncertainty_ns=10_000_000));self.assertEqual(x['coupling_need_uncertainty_range_ms'],[40,60])
 def test_unknown_binding_no_savings(self):
  x=classify(record(wait_binding='unknown'));self.assertIsNone(x['hypothetical_coupling_without_new_cost_ms']);self.assertIsNone(x['coupling_need_uncertainty_range_ms'])
 def test_current_binding_no_savings(self):
  x=classify(record(wait_binding='current_layer'));self.assertIsNone(x['hypothetical_coupling_without_new_cost_ms'])
 def test_unconfirmed_clock(self):
  r=record();del r['clock_alignment_verified'];self.assertEqual(classify(r)['status'],'unverified')
 def test_evidence_required(self):
  with self.assertRaises(ValueError):analyze({'records':[record()]})
 def test_no_rank_sum(self):
  x=analyze({'evidence_kind':'synthetic','records':[record(),record(rank=1)]});self.assertNotIn('total_gain',x);self.assertEqual(x['evidence_kind'],'synthetic')

if __name__=='__main__':unittest.main(verbosity=2)
