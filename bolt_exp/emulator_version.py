"""Deduce which version of an emulator a run used, for recording in results json.

Two sources are tried per repo, in order:

  1. The `hf_revision*` pin on the bolt problem class, when there is one. It is
     the revision the run actually downloaded, so it stays correct even when
     `main` has moved on since.
  2. The HF hub repo: whichever tag (if any) currently points at the commit
     `main` resolves to, via the huggingface_hub API. Only meaningful for
     unpinned repos, which track `main` by definition.

Either source can come up empty (no pin, no tag on main, huggingface_hub not
installed, no network) -- that's not an error, the version for that repo is
just left out of the result.
"""

# (repo attribute, revision attribute) pairs used across bolt.problems.* classes.
_REPO_ATTRS = (
    ("hf_repo", "hf_revision"),
    ("hf_repo_high_fid", "hf_revision_high_fid"),
    ("hf_repo_low_fid", "hf_revision_low_fid"),
    ("hf_repo_noise", "hf_revision_noise"),
)


def _hub_tag(repo_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None

    api = HfApi()
    for repo_type in ("model", "dataset"):
        try:
            refs = api.list_repo_refs(repo_id, repo_type=repo_type)
        except Exception:
            continue

        main_sha = next(
            (b.target_commit for b in refs.branches if b.name == "main"), None
        )
        if main_sha is None:
            return None

        for tag in refs.tags:
            # Annotated tags report the tag object's sha, not the commit's
            try:
                if api.repo_info(
                    repo_id, repo_type=repo_type, revision=tag.name
                ).sha == main_sha:
                    return tag.name
            except Exception:
                continue
        return None
    return None


def emulator_version(repo_id: str, revision: str | None = None) -> str | None:
    """Best-effort version label for `repo_id`: the pinned revision, else a hub tag on main."""
    return revision or _hub_tag(repo_id)


def emulator_versions(repo_ids) -> dict[str, str]:
    """{repo_id: version} for every repo a version could be deduced for."""
    versions = {}
    for repo_id in repo_ids:
        version = emulator_version(repo_id)
        if version is not None:
            versions[repo_id] = version
    return versions


def emulator_versions_for(prob) -> dict[str, str]:
    """Convenience: pull every hf_repo* attribute off a bolt problem instance."""
    pinned = {}
    for repo_attr, rev_attr in _REPO_ATTRS:
        repo_id = getattr(prob, repo_attr, None)
        if repo_id is None:
            continue
        # First pin wins; one repo can appear at two fidelities
        pinned.setdefault(repo_id, getattr(prob, rev_attr, None))

    versions = {}
    for repo_id in sorted(pinned):
        version = emulator_version(repo_id, pinned[repo_id])
        if version is not None:
            versions[repo_id] = version
    return versions
