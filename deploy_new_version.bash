# increase the version in the pyproject.toml file

# Check installation works:
pip uninstall woodelf_explainer
pip install -e .

python -c "import woodelf_explainer; print(woodelf_explainer.__version__)"

# Build the package
python -m pip install --upgrade build twine
python -m build
# see files were created in dist/

# Upload to pypi
python -m twine upload dist/*

# check installation
pip uninstall woodelf_explainer
pip install woodelf-explainer
python -c "import woodelf_explainer; print(woodelf_explainer.__version__)"

# Add the version to git
X.Y.Z should be replaced with the new version
git tag vX.Y.Z
git push origin vX.Y.Z