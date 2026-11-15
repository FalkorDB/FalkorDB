import FalkorTemporal.EraFwd0
import FalkorTemporal.EraBwd0
import FalkorTemporal.EraFwd1
import FalkorTemporal.EraBwd1
import FalkorTemporal.EraFwd2
import FalkorTemporal.EraBwd2
import FalkorTemporal.EraFwd3
import FalkorTemporal.EraBwd3
namespace FalkorTemporal

theorem fwdOk_all (k : Nat) (hk : k < 146097) : fwdOk k = true := by
  if h0 : k < 10000 then exact allRange_spec fwdOk 0 10000 fwd_0 k (by omega) (by omega) else
  if h10000 : k < 20000 then exact allRange_spec fwdOk 10000 10000 fwd_10000 k (by omega) (by omega) else
  if h20000 : k < 30000 then exact allRange_spec fwdOk 20000 10000 fwd_20000 k (by omega) (by omega) else
  if h30000 : k < 40000 then exact allRange_spec fwdOk 30000 10000 fwd_30000 k (by omega) (by omega) else
  if h40000 : k < 50000 then exact allRange_spec fwdOk 40000 10000 fwd_40000 k (by omega) (by omega) else
  if h50000 : k < 60000 then exact allRange_spec fwdOk 50000 10000 fwd_50000 k (by omega) (by omega) else
  if h60000 : k < 70000 then exact allRange_spec fwdOk 60000 10000 fwd_60000 k (by omega) (by omega) else
  if h70000 : k < 80000 then exact allRange_spec fwdOk 70000 10000 fwd_70000 k (by omega) (by omega) else
  if h80000 : k < 90000 then exact allRange_spec fwdOk 80000 10000 fwd_80000 k (by omega) (by omega) else
  if h90000 : k < 100000 then exact allRange_spec fwdOk 90000 10000 fwd_90000 k (by omega) (by omega) else
  if h100000 : k < 110000 then exact allRange_spec fwdOk 100000 10000 fwd_100000 k (by omega) (by omega) else
  if h110000 : k < 120000 then exact allRange_spec fwdOk 110000 10000 fwd_110000 k (by omega) (by omega) else
  if h120000 : k < 130000 then exact allRange_spec fwdOk 120000 10000 fwd_120000 k (by omega) (by omega) else
  if h130000 : k < 140000 then exact allRange_spec fwdOk 130000 10000 fwd_130000 k (by omega) (by omega) else
  if h140000 : k < 146097 then exact allRange_spec fwdOk 140000 6097 fwd_140000 k (by omega) (by omega) else
  omega

theorem bwdOk_all (k : Nat) (hk : k < 148800) : bwdOk k = true := by
  if h0 : k < 10000 then exact allRange_spec bwdOk 0 10000 bwd_0 k (by omega) (by omega) else
  if h10000 : k < 20000 then exact allRange_spec bwdOk 10000 10000 bwd_10000 k (by omega) (by omega) else
  if h20000 : k < 30000 then exact allRange_spec bwdOk 20000 10000 bwd_20000 k (by omega) (by omega) else
  if h30000 : k < 40000 then exact allRange_spec bwdOk 30000 10000 bwd_30000 k (by omega) (by omega) else
  if h40000 : k < 50000 then exact allRange_spec bwdOk 40000 10000 bwd_40000 k (by omega) (by omega) else
  if h50000 : k < 60000 then exact allRange_spec bwdOk 50000 10000 bwd_50000 k (by omega) (by omega) else
  if h60000 : k < 70000 then exact allRange_spec bwdOk 60000 10000 bwd_60000 k (by omega) (by omega) else
  if h70000 : k < 80000 then exact allRange_spec bwdOk 70000 10000 bwd_70000 k (by omega) (by omega) else
  if h80000 : k < 90000 then exact allRange_spec bwdOk 80000 10000 bwd_80000 k (by omega) (by omega) else
  if h90000 : k < 100000 then exact allRange_spec bwdOk 90000 10000 bwd_90000 k (by omega) (by omega) else
  if h100000 : k < 110000 then exact allRange_spec bwdOk 100000 10000 bwd_100000 k (by omega) (by omega) else
  if h110000 : k < 120000 then exact allRange_spec bwdOk 110000 10000 bwd_110000 k (by omega) (by omega) else
  if h120000 : k < 130000 then exact allRange_spec bwdOk 120000 10000 bwd_120000 k (by omega) (by omega) else
  if h130000 : k < 140000 then exact allRange_spec bwdOk 130000 10000 bwd_130000 k (by omega) (by omega) else
  if h140000 : k < 148800 then exact allRange_spec bwdOk 140000 8800 bwd_140000 k (by omega) (by omega) else
  omega

end FalkorTemporal
