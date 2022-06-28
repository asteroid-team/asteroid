import os
import torch
import subprocess
from pprint import pprint


PLEASE_PUBLISH = (
    "\nDon't forget to share your pretrained models at "
    "https://zenodo.org/communities/asteroid-models/! =)\n"
    "You can directly use our CLI for that, run this: \n"
    '`asteroid-upload {} --uploader "Your name here"`\n'
)

HREF = '<a href="{}">{}</a>'
CC_SA = "Attribution-ShareAlike 3.0 Unported"
CC_SA_LINK = "https://creativecommons.org/licenses/by-sa/3.0/"
ASTEROID_REF = HREF.format("https://github.com/asteroid-team/asteroid", "Asteroid")


def save_publishable(publish_dir, model_dict, metrics=None, train_conf=None, recipe=None):
    """Save models to prepare for publication / model sharing.

    Args:
        publish_dir (str): Path to the publishing directory.
            Usually under exp/exp_name/publish_dir
        model_dict (dict): dict at least with keys `model_args`,
            `state_dict`,`dataset` or `licenses`
        metrics (dict): dict with evaluation metrics.
        train_conf (dict): Training configuration dict (from conf.yml).
        recipe (str): Name of the recipe.

    Returns:
        dict, same as `model_dict` with added fields.

    Raises:
        AssertionError when either `model_args`, `state_dict`,`dataset` or
            `licenses` are not present is `model_dict.keys()`
    """
    assert "model_args" in model_dict.keys(), "`model_args` not found in model dict."
    assert "state_dict" in model_dict.keys(), "`state_dict` not found in model dict."
    assert "dataset" in model_dict.keys(), "`dataset` not found in model dict."
    assert "licenses" in model_dict.keys(), "`licenses` not found in model dict."
    assert isinstance(metrics, dict), "Cannot upload a model without metrics."
    # Additional infos.
    if recipe is not None:
        assert isinstance(recipe, str), "`recipe` should be a string."
        recipe_name = recipe
    else:
        if os.path.exists(os.path.join(publish_dir, "recipe_name.txt")):
            recipe_name = next(open(os.path.join(publish_dir, "recipe_name.txt")))
            recipe_name.replace("\n", "")  # remove next line
        else:
            recipe_name = "Unknown"
    model_dict["infos"]["recipe_name"] = recipe_name
    model_dict["infos"]["training_config"] = train_conf
    model_dict["infos"]["final_metrics"] = metrics
    os.makedirs(publish_dir, exist_ok=True)
    torch.save(model_dict, os.path.join(publish_dir, "model.pth"))
    print(PLEASE_PUBLISH.format(publish_dir))
    return model_dict


def upload_publishable(
    publish_dir,
    uploader=None,
    affiliation=None,
    git_username=None,
    token=None,
    force_publish=False,
    use_sandbox=False,
    unit_test=False,
):
    """Entry point to upload publishable model.

    Args:
        publish_dir (str): Path to the publishing directory.
            Usually under exp/exp_name/publish_dir
        uploader (str): Full name of the uploader (Ex: Manuel Pariente)
        affiliation (str, optional): Affiliation (no accent).
        git_username (str, optional): GitHub username.
        token (str): Access token generated to upload depositions.
        force_publish (bool): Whether to directly publish without
            asking confirmation before. Defaults to False.
        use_sandbox (bool): Whether to use Zenodo's sandbox instead of
            the official Zenodo.
        unit_test (bool): If True, we do not ask user input and do not publish.

    """

    def get_answer():
        out = input("\n\nDo you want to publish it now (irreversible)? y/n" "(Recommended: n).\n")
        if out not in ["y", "n"]:
            print(f"\nExpected one of [`y`, `n`], received {out}, please retry.")
            return get_answer()
        return out

    if uploader is None:
        raise ValueError("Need uploader name")

    # Make publishable model and save it
    model_path = os.path.join(publish_dir, "model.pth")
    publish_model_path = os.path.join(publish_dir, "published_model.pth")
    model = torch.load(model_path)
    model = _populate_publishable(
        model,
        uploader=uploader,
        affiliation=affiliation,
        git_username=git_username,
    )
    torch.save(model, publish_model_path)

    # Get Zenodo access token
    if token is None:
        token = os.getenv("ACCESS_TOKEN")
        if token is None:
            raise ValueError(
                "Need an access token to Zenodo to upload the model. Either "
                "set ACCESS_TOKEN environment variable or pass it directly "
                "(`asteroid-upload --token ...`)."
                "If you do not have a access token, first create a Zenodo "
                "account (https://zenodo.org/signup/), create a token "
                "https://zenodo.org/account/settings/applications/tokens/new/"
                "and you are all set to help us! =)"
            )

    # Do the actual upload
    zen, dep_id = zenodo_upload(
        model, token, model_path=publish_model_path, use_sandbox=use_sandbox
    )
    address = os.path.join(zen.zenodo_address, "deposit", str(dep_id))
    if force_publish:
        r_publish = zen.publish_deposition(dep_id)
        pprint(r_publish.json())
        print("You can also visit it at {}".format(address))
        return r_publish
    # Give choice
    current = zen.get_deposition(dep_id)
    print(f"\n\n This is the current state of the deposition " f"(see here {address}): ")
    pprint(current.json())
    # Patch to run unit test
    if unit_test:
        return zen, current
    else:
        inp = get_answer()
    # Get user input
    if inp == "y":
        _ = zen.publish_deposition(dep_id)
        print("Visit it at {}".format(address))
    else:
        print(f"Did not finalize the upload, please visit {address} to finalize " f"it.")


def _populate_publishable(model, uploader=None, affiliation=None, git_username=None):
    """Populate infos in publishable model.

    Args:
        model (dict): Model to publish, with `infos` key, at least.
        uploader (str): Full name of the uploader (Ex: Manuel Pariente)
        affiliation (str, optional): Affiliation (no accent).
        git_username (str, optional): GitHub username.

    Returns:
        dict (model), same as input `model`

    .. note:: If a `git_username` is not specified, we look for it somehow, or take
        the laptop username.
    """
    # Get username somehow
    if git_username is None:
        git_username = get_username()

    # Example: mpariente/ConvTasNet_WHAM_sepclean
    model_name = "_".join([model["model_name"], model["dataset"], model["task"].replace("_", "")])
    upload_name = git_username + "/" + model_name
    # Write License Notice
    license_note = make_license_notice(model_name, model["licenses"], uploader=uploader)
    # Add infos
    model["infos"]["uploader"] = uploader
    model["infos"]["git_username"] = git_username
    model["infos"]["affiliation"] = affiliation if affiliation else "Unknown"
    model["infos"]["upload_name"] = upload_name
    model["infos"]["license_note"] = license_note
    return model


def get_username():
    """Get git of FS username for upload."""
    username = subprocess.check_output(["git", "config", "user.name"])
    username = username.decode("utf-8")[:-1]
    if not username:  # Empty string
        import getpass

        username = getpass.getuser()
    return username


def make_license_notice(model_name, licenses, uploader=None):
    """Make license notice based on license dicts.

    Args:
        model_name (str): Name of the model.
        licenses (List[dict]): List of dict with
            keys (`title`, `title_link`, `author`, `author_link`,
                  `licence`, `licence_link`).
        uploader (str): Name of the uploader such as "Manuel Pariente".

    Returns:
        str, the license note describing the model, it's attribution,
            the original licenses, what we license it under and the licensor.
    """
    if uploader is None:
        raise ValueError("Cannot share model without uploader.")
    note = 'This work "{}" is a derivative '.format(model_name)
    for l_dict in licenses:
        # Clickable links in HTML.
        title = HREF.format(l_dict["title_link"], l_dict["title"])
        author = HREF.format(l_dict["author_link"], l_dict["author"])
        license_h = HREF.format(l_dict["license_link"], l_dict["license"])
        comm = " (Research only)" if l_dict["non_commercial"] else ""
        note += f"of {title} by {author}, used under {license_h}{comm}"
        note += "; "
    note = note[:-2] + ". "  # Remove the last ;
    cc_sa = HREF.format(CC_SA_LINK, CC_SA)
    note += f'"{model_name}" is licensed under {cc_sa} by {uploader}.'
    return note


def zenodo_upload(model, token, model_path=None, use_sandbox=False):
    """Create deposit and upload metadata + model

    Args:
        model (dict):
        token (str): Access token.
        model_path (str): Saved model path.
        use_sandbox (bool): Whether to use Zenodo's sandbox instead of
            the official Zenodo.

    Returns:
        Zenodo (Zenodo instance with access token)
        int (deposit ID)

    .. note::If `model_path` is not specified, save the model in tmp.pth and
        remove it after upload.
    """
    model_path_was_none = False
    if model_path is None:
        model_path_was_none = True
        model_path = "tmp.pth"
        torch.save(model, model_path)

    from .zenodo import Zenodo

    zen = Zenodo(token, use_sandbox=use_sandbox)
    metadata = make_metadata_from_model(model)
    r = zen.create_new_deposition(metadata=metadata)
    if r.status_code != 200:
        print(r.json())
        raise RuntimeError("Could not create the deposition, check the " "provided token.")
    dep_id = r.json()["id"]
    _ = zen.upload_new_file_to_deposition(dep_id, model_path, name="model.pth")
    if model_path_was_none:
        os.remove(model_path)
    return zen, dep_id


def make_metadata_from_model(model):
    """Create Zenodo deposit metadata for a given publishable model.

    Args:
        model (dict): Dictionary with all infos needed to publish.
            More info to come.

    Returns:
        dict, the metadata to create the Zenodo deposit with.

    .. note:: We remove the PESQ from the final results as a license is needed to
        use it.
    """
    infos = model["infos"]
    # Description section
    description = "<p><strong>Description: </strong></p>"
    tmp = "This model was trained by {} using the {} recipe in {}. "
    description += tmp.format(infos["uploader"], infos["recipe_name"], ASTEROID_REF)
    tmp = "</a>It was trained on the <code>{}</code> task of the {} dataset.</p>"
    description += tmp.format(model["task"], model["dataset"])

    # Training config section
    description += "<p>&nbsp;</p>"
    description += "<p><strong>Training config:</strong></p>"
    description += two_level_dict_html(infos["training_config"])

    # Results section
    description += "<p>&nbsp;</p>"
    description += "<p><strong>Results:</strong></p>"
    display_result = {k: v for k, v in infos["final_metrics"].items() if "pesq" not in k.lower()}
    description += display_one_level_dict(display_result)

    # Software section
    description += "<p>&nbsp;</p>"
    description += "<p><strong>Versions:</strong></p>"
    description += display_one_level_dict(infos["software_versions"])

    # License section
    description += "<p>&nbsp;</p>"
    description += "<p><strong>License notice:</strong></p>"
    description += infos["license_note"]

    # Putting it together.
    metadata = {
        "title": infos["upload_name"],
        "upload_type": "software",
        "description": description,
        "creators": [{"name": infos["uploader"], "affiliation": infos["affiliation"]}],
        "communities": [{"identifier": "zenodo"}, {"identifier": "asteroid-models"}],
        "keywords": [
            "Asteroid",
            "audio source separation",
            model["dataset"],
            model["task"],
            model["model_name"],
            "pretrained model",
        ],
        "license": "CC-BY-SA-3.0",
    }
    return metadata


def two_level_dict_html(dic):
    """Two-level dict to HTML.

    Args:
        dic (dict): two-level dict

    Returns:
        str for HTML-encoded two level dic
    """
    html = "<ul>"
    for k in dic.keys():
        # Open field
        html += f"<li>{k}: <ul>"
        for k2 in dic[k].keys():
            val = str(dic[k][k2])
            html += f"<li>{k2}: {val}</li>"
        # Close field
        html += "</il></ul>"
    html += "</ul>"
    return html


def display_one_level_dict(dic):
    """Single level dict to HTML

    Args:
        dic (dict):

    Returns:
        str for HTML-encoded single level dic
    """
    html = "<ul>"
    for k in dic.keys():
        # Open field
        val = str(dic[k])
        html += f"<li>{k}: {val} </li>"
    html += "</ul>"
    return html
