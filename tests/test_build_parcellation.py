"""The parcellation's pure-numpy pieces: the node table (84 nodes, 14 lobar groups), the majority vote and the
derived labels / stop mask. No SynthSeg, no TensorFlow, no download -- everything here is synthetic and fast."""
import numpy as np
import pytest

from tools import build_parcellation as BP


def test_node_table_has_84_nodes_and_14_lobar_groups():
    nodes = BP.node_table()
    assert len(nodes) == 84
    assert len(set(n["id"] for n in nodes)) == 84
    assert set(n["id"] for n in nodes) == set(range(1, 85))
    left = [n for n in nodes if n["hemisphere"] == "left"]
    right = [n for n in nodes if n["hemisphere"] == "right"]
    assert len(left) == 42 and len(right) == 42
    cortical = [n for n in nodes if n["lobe"] != "subcortical"]
    subcortical = [n for n in nodes if n["lobe"] == "subcortical"]
    assert len(cortical) == 68 and len(subcortical) == 16
    regions = BP.regions_payload()
    groups = regions["lobes"]
    assert len(groups) == 14 and len(set(groups)) == 14
    for lobe in BP.LOBES:
        assert f"{lobe}-left" in groups and f"{lobe}-right" in groups


def test_node_table_ids_are_unique_per_fs_label_and_hemisphere():
    nodes = BP.node_table()
    keys = [(n["fs_label"], n["hemisphere"]) for n in nodes]
    assert len(set(keys)) == len(keys)


def test_supersampled_affine_centres_subvoxels_within_the_parent_voxel():
    ref_affine = np.array([[2.5, 0, 0, 10.0], [0, 2.5, 0, -5.0], [0, 0, 2.5, 3.0], [0, 0, 0, 1]])
    for factor in (1, 2, 4, 5):
        sup = BP._supersampled_affine(ref_affine, factor)
        # the mean of the factor**3 sub-voxel centres of parent voxel (2, 1, 0) must equal that parent's own centre
        idx = np.arange(factor)
        grid = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), axis=-1).reshape(-1, 3)
        sub_index = np.array([2, 1, 0]) * factor + grid
        world = sub_index @ sup[:3, :3].T + sup[:3, 3]
        parent_world = ref_affine[:3, :3] @ [2, 1, 0] + ref_affine[:3, 3]
        np.testing.assert_allclose(world.mean(axis=0), parent_world, atol=1e-9)


def test_majority_vote_picks_the_plurality_label_under_identity_registration():
    # a tiny "T1" label volume at 1 mm, two blocks of 3x3x3 mostly-9 voxels with a 2-voxel minority, at the origin;
    # the "diffusion" grid is 3x coarser (3 mm) and exactly covers one combined super-voxel.
    seg = np.full((3, 3, 3), 9, dtype=np.int32)
    seg[0, 0, 0] = 5
    seg[0, 0, 1] = 5
    seg_affine = np.eye(4)
    ref_affine = np.diag([3.0, 3.0, 3.0, 1.0])
    ref_affine[:3, 3] = 1.0  # the single ref voxel's centre sits at the seg block's centre
    majority = BP.majority_vote(seg, seg_affine, (1, 1, 1), ref_affine, np.eye(4), factor=3)
    assert majority.shape == (1, 1, 1)
    assert majority[0, 0, 0] == 9  # 25 votes for 9 against 2 for 5


def test_labels_and_stop_mask_from_majority_labels():
    majority = np.array([[[0, 2, 1001]]])  # background, left cerebral WM, left bankssts
    labels, stop = BP.labels_and_stop_mask(majority)
    assert labels[0, 0, 0] == 0
    assert labels[0, 0, 2] == 1  # bankssts is fs id 1001, the first (lowest) left-hemisphere node
    assert stop.tolist() == [[[False, True, False]]]


def test_label_volumes_mm3():
    labels = np.array([0, 0, 1, 1, 1, 2], dtype=np.int16)
    vols = BP.label_volumes_mm3(labels, voxel_volume_mm3=8.0)
    assert vols == {1: 24.0, 2: 8.0}


def test_every_cortical_name_has_a_lobe():
    assert set(BP.LOBE_OF_CORTICAL) == set(BP.CORTICAL_NAMES.values())
    assert set(BP.LOBE_OF_CORTICAL.values()) <= set(BP.LOBES)
