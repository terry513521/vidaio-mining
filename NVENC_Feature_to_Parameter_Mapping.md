# Video Features → NVENC Parameter Mapping

## Goal

Use measurable video features to choose NVENC parameters, while
searching only **CQ** (and optionally **Preset**).

**Hard rule:** features never set or bias CQ. CQ is found only by
measured search (maximize `s_f` under the VMAF gate). Features only
rank which non-CQ NVENC knobs Round 2 should try.

------------------------------------------------------------------------

# 1. Motion

### Features

-   Mean optical flow
-   95th percentile motion
-   Camera motion
-   Motion variance

### Effect

  Motion   Recommended
  -------- -----------------------------------------------------------
  Low      Higher CQ, lower lookahead, fewer B-frames
  Medium   Balanced settings
  High     Lower CQ, higher lookahead, Temporal AQ ON, more B-frames

**Reason:** High motion makes inter prediction harder and usually
requires more bits.

------------------------------------------------------------------------

# 2. Texture

### Features

-   Edge density
-   Laplacian variance
-   High-frequency energy
-   Entropy

### Effect

  Texture   Recommended
  --------- --------------------------------------
  Low       Higher CQ, lower AQ strength
  Medium    Default AQ
  High      Lower CQ, Spatial AQ ON, stronger AQ

**Reason:** Fine textures (grass, trees, brick walls) are expensive to
compress.

------------------------------------------------------------------------

# 3. Noise / Film Grain

### Features

-   Noise variance
-   Temporal noise
-   Grain estimate

### Effect

  Noise    Recommended
  -------- ------------------------------------------------
  Low      Default AQ
  Medium   Moderate AQ
  High     Lower AQ strength, avoid over-preserving grain

**Reason:** Random noise consumes many bits but is not very predictable.

------------------------------------------------------------------------

# 4. Scene Changes

### Features

-   Scene-cut count
-   Average scene duration

### Effect

  Scene changes   Recommended
  --------------- -----------------------------
  Few             Longer GOP
  Many            Shorter GOP, more lookahead

**Reason:** Frequent scene cuts reduce prediction efficiency.

------------------------------------------------------------------------

# 5. Brightness

### Features

-   Mean luminance
-   Histogram

### Effect

  Brightness   Recommended
  ------------ --------------------
  Dark         Slightly lower CQ
  Bright       Slightly higher CQ

**Reason:** Dark scenes often contain sensor noise and subtle gradients.

------------------------------------------------------------------------

# 6. Contrast

### Features

-   Dynamic range
-   Luma variance

### Effect

High contrast generally benefits from Spatial AQ.

------------------------------------------------------------------------

# 7. Color Complexity

### Features

-   Chroma variance
-   Saturation

### Effect

Highly saturated videos may require slightly lower CQ to avoid chroma
artifacts.

------------------------------------------------------------------------

# 8. Resolution

### Effect

  Resolution   Recommended
  ------------ -----------------------
  720p         Higher CQ
  1080p        Medium CQ
  4K           Lower CQ, stronger AQ

------------------------------------------------------------------------

# 9. Frame Rate

  FPS      Recommendation
  -------- -------------------------------
  24--30   Default
  60+      More lookahead, more B-frames

------------------------------------------------------------------------

# Recommended Rule Engine

  --------------------------------------------------------------------------------------
  Feature             CQ Preset   Spatial   Temporal           AQ Lookahead     B-frames
                                  AQ        AQ           Strength             
  ----------- ---------- -------- --------- ---------- ---------- ----------- ----------
  High motion          ↓ p6--p7   ON        ON              8--12 High              4--5

  High                 ↓ p6--p7   ON        Optional       10--12 Medium            3--4
  texture                                                                     

  Heavy noise          ↓ p5--p6   ON        OFF/Low          4--6 Medium               3

  Static               ↑ p4--p5   OFF       OFF                 4 Low                  2
  talking                                                                     
  head                                                                        

  Screen               ↑ p5       ON        OFF                 8 Low                  2
  recording                                                                   

  Anime                ↑ p5       ON        OFF             8--10 Low                  3
  --------------------------------------------------------------------------------------

Legend: - Higher CQ = smaller file / lower quality. - Lower CQ = larger
file / higher quality.

# Suggested Optimization Pipeline

1.  Extract video features (detail-scale noise/texture, per-frame motion).
2.  **Do not set CQ from features.** CQ is found by search only.
3.  Apply features → **NVENC baseline** before Round 1:
    -   Spatial / Temporal AQ
    -   AQ strength (noise wins over texture)
    -   Lookahead / B-frames / GOP / b_ref_mode
4.  Round 1: linspace CQ search with that baseline.
5.  Round 2: lock best CQ → try top feature-ranked NVENC variants →
    refine nearby CQs.
6.  Measure VMAF / s_f and keep the best trial.

Legend: Higher CQ = smaller file / lower quality. Lower CQ = larger
file / higher quality. For s_f maximization, search wants the highest CQ
that still clears the VMAF gate.
