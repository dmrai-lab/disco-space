# The scanner on the two pages

A replay pack stores where the walkers went. It does not store the machine. The paper's replay equation lets every
scanner term in at replay time ("Where replay ends", Table `tab:scanner`). These are the terms the pages apply, where
each one enters on each page, how exact it is, what it costs, and what is refused.

The scanner menu has four entries: **ideal** (no scanner terms, any gradient amplitude, the field preset and
direction free) and three catalogued machines, the **Hyperfine Swoop 64 mT**, the **Siemens Prisma 3 T** and the
**Siemens Terra 7 T** (`dmipy_sim.acquisition.scanners.ScannerLimits`). Choosing a machine fixes the field strength
and its direction to the catalogue's (the Swoop's field runs anterior, across the patient, and the cylinders' runs
along the bore). Its gradient limit refuses a shell it cannot play, and the page applies every term the catalogue
carries for that machine and nothing else. A term the catalogue does not carry is absent and the page says so. It is
never filled in.

## The terms

Write a walker's position as the voxel's place plus its own excursion, `x_v + u_w(t)`, and expand the field across
the voxel (paper Eq. `eq:voxel`).

| term | order | Swoop | Prisma / Terra | DiSCo page (moment layout) | brain page (free waveform) |
|---|---|---|---|---|---|
| field law value, drift, Maxwell field value | 0: one phase per voxel | law measured (Z2 derived from the measured homogeneity and gradient); drift measured | no shape published; none | **no effect on a magnitude image**; not computed | same |
| gradient nonlinearity `L(x_v)` | 1 | measured diagonal, off-diagonals derived (minimum-norm member) | class model inferred (Malyarenko 2016 / Rogers 2017), isotropic scale | `q -> a L u` per voxel and measurement: **exact** | per encoding class: **exact for the class representative**, voxels binned to 1 % of b |
| Maxwell (concomitant) gradient `grad Bc` | 1 | exact `|B|` law from the field strength and the coils | same | `+ gc(a L u)` per voxel and measurement: **exact**, because a square-pulse shape is two-valued (derived below) | in the class's played waveform; ramps leave a rank-2 residual, bounded and added to the misfit (3e-6 at the Prisma) |
| magnet's own gradient `g0(x_v)` | 1 | the law's gradient (1.4 mT/m at 8 cm) | none published: absent | `+ g0 . n_w` with `n_w` the walker's **background moment**, one more column per sequence group: **exact** | **Swoop refused on the brain page** (below) |
| head's own field | 0 and 1 | from the phantom, not the catalogue | | DiSCo is a 1 mm cube with no head: absent | not modelled (the brain page already says so) |
| transmit scale `kappa(x_v)` | RF | measured axial fall-off with its harmonic transverse part, measured calibration offset | catalogue holds a brain range, no map: absent | a per-voxel factor on the crushed echo's pathway amplitude, `eta(kappa alpha) / eta(alpha)`: **exact** | Swoop refused; absent for the cylinders |
| spoiler / crusher voxel factor `V(k)` | 1 | | | the pages play balanced PGSE and stimulated echoes, so the factor is 1 | same |
| field strength and direction | scalars of C3 | 0.064 T along A | 3 / 7 T along S | the susceptibility tier's `B0`, `b0_direction` | the specimen pose's field |

Order two (field curvature across a voxel) is not claimed, and the sum over voxels is image formation. Neither is
replay.

**Where the catalogue's labels come from.** Every number on the page is a catalogue leaf with a confidence
(`scanner_constants.json`). The page shows *cited* as **measured**, *derived* as **derived from a measurement**,
*inferred* as **inferred from the class**, and an absent leaf as **absent**. Each machine's term table on the page is
built from those leaves when the menu changes.

## Placement

- **DiSCo** is 1 mm across, so where it sits decides everything. At isocentre every term vanishes (the shimmed law's
  gradient is zero there, the nonlinearity is the identity and the Maxwell term is zero). The page offers a
  **distance from isocentre** (0 to 7.9 cm) along one scanner axis (R-L, A-P or S-I), and the phantom's grid is
  `Grid.centred_at(offset)`, with every voxel's terms evaluated at that voxel. 7.9 cm keeps the whole cube inside the
  Swoop's 8 cm anchor. Beyond the anchor the law is refused, not extrapolated.
- **The brain** has its head centre (the brain mask's centroid) at isocentre, with the image's own orientation in the
  scanner. The bore grid is `Grid.from_oblique_affine` of the asset's affine on the voxels' own indices
  (`Brain.bore_grid`). dmipy-sim places a left-handed grid where its affine puts it (dmipy-sim#572), so every voxel's
  offset is its physical one. The asset's affine is left-handed; a test holds the Swoop's field offset, transmit scale
  and own gradient and the Prisma's nonlinearity on a left-handed copy of the fixture to the catalogue's law at
  `affine @ ijk`. The head reaches 8.3 cm from its centre (196 of 71,052 voxels lie beyond 8 cm), well inside the
  Prisma and Terra models' 11.3 cm.

## The two routes are different, and so is the cost

**DiSCo runs on moments.** The hosted DiSCo Space replays from the shape-moment layout. Per stored timing class
there is one 3-vector moment per walker, and a measurement is the phase `g u . m_w`. Only those classes can be
played. A machine enters by replacing `g u` with the delivered vector and adding the background term:

    phase_vw,i = q_v,i . m_w + g0_v . n_w,      q_v,i = a_i L_v u_i + gc(a_i L_v u_i; x_v)

- *Nonlinearity:* the coils deliver `L G(t) = a s(t) L u`. This keeps the shape and changes the vector. It is exact
  because the band projection is linear in the waveform.
- *Maxwell gradient:* `gc` is a pointwise function of the coils' gradient at each instant
  (`scanner_sequence.concomitant_terms`, the exact `|B|` law, not its leading order) and vanishes with it. A
  square-pulse PGSE or PGSTE plays both lobes at one polarity, so `s(t)` takes only the values 0 and 1, and therefore
  `gc(L G(t)) = s(t) gc(a L u)`: one more vector on the same moment. With ramps this would be `s^2`-shaped and not one
  vector, and `delivered_moments` refuses it by name. The layout's classes are square.
- *Background:* the magnet's gradient is constant and on through the pulses and dead times. Its effective waveform is
  the coherence sign times the transverse gate, `eps(t) chi(t) g0`. That is one more time course per **sequence
  group** (the classes sharing a grid, an RF schedule and a readout). DiSCo has two groups, the six PGSE classes at
  TE 53.5 ms and the stimulated echo, so two `(n_tiles, 128, 3)` float32 columns of 1.9 GB each, written in one pass
  over the columnar pack by `dmipy_sim.replay.shape_moments.stamp_background` without touching any other column.
  The layout manifest's stamp record: one pass of 42.4 GB from the Hub, 1,016 s on gaia's CPU. They are published at
  `SubstrateCommons/disco-replay` revision 2db4e78, the revision `config.toml` pins.
- *Transmit:* each voxel's signal is multiplied by the crushed echo's pathway amplitude at its scale over the nominal
  one (`epg.transmit_amplitude`): `sin^3(90 kappa)` for the spin echo, `0.5 sin^3(90 kappa)` over `0.5` for the
  stimulated echo.

dmipy-sim#560 holds this against the reference route at 1e-7, for each machine with every term on: each voxel's
acquisition rebuilt through `ScannerSequence.with_gradient_nonlinearity / with_background_gradient /
with_concomitant` and replayed walker by walker. The cost per run is one `(n_voxels, n_meas, 3)` array of delivered
vectors on the host (`bore.delivered_moments`, closed forms only) and a per-tile gather in the kernel.

**The brain is free.** The brain Space is a compositional phantom with three substrates. Its responses are computed
per run from the packs (the pose responses, `ReplayPack.pose_responses`) for whatever waveform the run plays. It has
no moments. A machine therefore enters as the **actual delivered waveform**. The voxels are binned into encoding
classes by what they receive (`bore.encoding_classes`: the nonlinearity tensor to half the tolerance, the Maxwell
term's position to `tol B0 / G_max`, the background to `tol/2` of the commanded gradient). One acquisition is
composed per class from a member's exact values (`ScannerSequence.with_*`) and expanded by the packs. Every voxel then
composes its own class's responses with its FOD and fractions. This is exactly `Phantom.compose(...).replay(seq,
scanner=ScannerLimits, encoding_tolerance=tol)`, which the brain's tests hold it to.

Measured at this pin on MASiVar's head (71,052 voxels, the scan's 485 measurements, the default packs: CACTUS 1 s in
125 ms windows, grey-matter spheres 250 ms; every tier on; `tools/measure_scanners.py`, JAX on gaia's GH200, the
contraction in torch on the same device, 2026-10-04):

| | encoding classes at 1 % of b | `prepare` (pose responses), s | of it WM / GM, s | replay (contraction) first / steady, s | image vs the ideal scanner at its field, M0 units, median / max |
|---|---|---|---|---|---|
| ideal at 3 T | 1 | 9.2 | | 0.10 / 0.04 | |
| Prisma 3 T | 6 | 167.9 | 157.2 / 9.4 | 0.05 / 0.05 | 1.9e-4 / 2.7e-3 |
| ideal at 7 T | 1 | 18.9 | | 0.04 / 0.04 | |
| Terra 7 T | 6 | 111.3 | 103.2 / 6.9 | 0.04 / 0.04 | 1.9e-4 / 2.7e-3 |

- **Classes:** the Prisma and the Terra deliver between -2.3 % and +2.7 % of the nominal b over the head (the
  delivered-b table below), which the 1 % tolerance bins into 6 classes each.
- **Cost per class:** 18-28 s of pose responses on the GH200, against 9-19 s for the whole commanded protocol on the
  ideal scanner. Every row of a class plays its own amplitude, so the shells no longer share a body. The contraction
  is unchanged. Not re-measured here: the CPU figures of the previous pin (`prepare` 354 s on the Prisma and 405 s on
  the Terra with single-threaded BLAS, on the 100 ms CACTUS pack, since deleted from the Hub).
- **Ramps:** the scanner's slew makes the Maxwell gradient a rank-2 waveform, which used to send every class to the
  quadrature at 35 s per measurement. The closed form takes its principal direction when the residual's phase, bounded
  over every pose, stays under a tenth of the floor, and adds the bound to the misfit (dmipy-sim#561, merged as #573).
- **Pricing:** the warm-up computes the scan protocol at each machine when the container starts, which is not charged
  to a visitor. A run of it then costs what the ideal scanner's does. A custom protocol at a machine is priced at its
  classes in the reservation (`response_per_class`). For the scan protocol that comes to the 480 s cap, which no
  visitor's quota covers (measured again here: the scan on either machine prices at 16,296 cold rows and reserves the
  480 s cap), so only what the warm-up cached is runnable on a machine.

## What is refused, and why

- **The Swoop on the brain page.** Its own gradient is on through every pulse and dead time, so every voxel plays a
  waveform with two directions (the encoding and `g0`) and two time courses. dmipy-sim's closed-form pose expansion
  now takes it exactly: the background gradient is a second plane-wave factor beside the field's, within a bounded
  residual, summed as a shell series (dmipy-sim#565, #573, #582, #588). What keeps the Swoop off the brain's menu is
  the cost, not the physics. Measured at this pin on MASiVar's head (71,052 voxels, the scan's 485 measurements, the
  Swoop's slew; the classes counted by `encoding_classes`' own binning without composing each one, since composing
  them all is dmipy-sim#563's memory bound, and three composed classes timed):
  - **Classes:** 196 voxels lie beyond the Swoop's 8 cm anchor, where its law is refused. The other 70,856 bin into
    **9,752 encoding classes** at the menu's 1 % of b, and 665 at 3 %. The background is binned to half the tolerance
    of the commanded gradient, and the Swoop's `g0` (up to 1.4 mT/m at 8 cm against a 23 mT/m commanded gradient)
    spans many such bins across a head. The earlier figure of 63 classes was a 1,312-voxel subsample at 3 %.
  - **Cost per class** (the default packs, every tier, the Swoop's field; JAX): WM 104-166 s and GM 4-21 s on gaia's
    GH200, three classes; WM 484-560 s and GM 51-52 s on gaia's CPU (8 threads, a shared box), two classes.
  - So one protocol at one tissue and field is about 9,752 x 2.5 min, two weeks of one GPU (a day at 3 %). The
    page shows this reason under its scanner menu (`Brain.refused_machines`). Whether to cache the default
    protocol's classes or to keep the Swoop to the DiSCo page is the owner's decision. On the DiSCo page the Swoop is
    exact.
- **A machine in DiSCo's full mode.** The columnar route replays one commanded sequence for the whole grid, and the
  scanner's per-voxel waveform is demo mode's. Full mode is not on the hosted Spaces.
- **A shape with ramps on the moment layout** (the Maxwell term is then not one vector) and a placement beyond a
  catalogued law's anchor. Both refusals name the reason.
- **The scanner's slew on DiSCo.** The layout stores square pulses, so a machine's slew is not played there. Its
  gradient limit still refuses what it cannot play. On DiSCo's TE 53.5 ms classes this leaves the Swoop (23 mT/m)
  b-values up to about 350 s/mm² (δ 17.7 / Δ 35.8 ms), and the Prisma and Terra (80 mT/m) the clinical and research
  presets but not DiSCo 364.

## Measured effects

All runs are noiseless and the reference is the ideal scanner at the machine's field and direction, so only the
delivered gradient and the transmit scale differ.

**DiSCo, phantom centre 7.9 cm from isocentre along R-L.** Re-measured at this pin on gaia's GH200 from the published
layout at revision 2db4e78 (`tools/scanner_disco_effects.py`, `LAYOUT=` its local copy), CSD order 8, 659,840 seeds;
every number below is the previous pin's to the digits shown.

| machine | protocol (what it can play) | connectome Pearson vs strand count, ideal -> machine | direction-mean shell change (median, 1-99 %) | per-measurement abs(dS), median / 99 % | S0 |
|---|---|---|---|---|---|
| Swoop 64 mT | b 350 x 60 on δ 17.7 / Δ 35.8 ms | **0.927 -> 0.650** | +0.01 % (-0.2, +0.3 %) | 0.0080 / 0.0177 (floor 0.0024) | 0.90 |
| Prisma 3 T | research 3-shell x 90 (b 1000 / 2000 / 3000) | 0.925 -> 0.924 | -1.0 / -1.7 / -2.3 % | 0.0066-0.0081 / 0.0081-0.0090 (floor 0.0074) | 1 |
| Terra 7 T | the same | 0.919 -> 0.917 | -0.9 / -1.7 / -2.2 % | the same | 1 |

The Swoop's terms one at a time on its shell (`tools/scanner_disco_terms.py`, re-measured the same way; per-measurement
abs(dS), median / 99 %):

| term | median | 99 % |
|---|---|---|
| background g0 | 0.0074 | 0.0171 |
| nonlinearity | 0.0040 | 0.0117 |
| Maxwell | 0.0002 | 0.0009 |
| transmit | 0 | 0 |

The transmit scale cancels in the S0-normalised DWI; it shows only in S0 (0.90). The background's cross term flips
with the direction, so the direction mean hides it (+0.02 %). The connectome does not hide it: the shell's angular
contrast at b 350 is of the same order as the term. The cylinders' class model is an isotropic scale, so it moves
every direction's b alike (+2.4 % at 7.9 cm transverse). The shells drop by percent while the FOD shape and the
connectome keep.

**The brain, head centre at isocentre.** Delivered b over the b = 1000 shell (96 directions, square pulses,
`tools/scanner_brain_bmap.py`, re-measured at this pin on the bore grid of the plain index):

| machine | direction-mean b / b (1 %, median, 99 %) | per direction (min, max) | transmit scale |
|---|---|---|---|
| Swoop (196 voxels beyond its 8 cm anchor excluded) | 0.983, 1.001, 1.021 | 0.83, 1.27 | 0.95-1.17 |
| Prisma | 0.982, 1.004, 1.021 | 0.977, 1.027 | not catalogued |
| Terra | 0.982, 1.004, 1.021 | 0.977, 1.027 | not catalogued |

On the brain page the Prisma and the Terra move the replayed image by a median 1.9e-4 and at most 2.7e-3 in M0
units (scan protocol, 6 classes each, the default packs; the table above).
