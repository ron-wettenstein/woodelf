# increase the version in the woodelf/__init__.py file

# Check installation works:
pip uninstall -y woodelf_explainer
pip install -e .

python -c "import os; os.chdir('..'); os.chdir('..'); import woodelf; print(woodelf); print(woodelf.__version__)"

# Build the package
#If needed: python -m pip install --upgrade build twine
windows: del .\dist\*
linux: rm .\dist\*

python -m build
# see files were created in dist/

# Upload to pypi
python -m twine upload dist/*

# check installation
pip uninstall -y woodelf_explainer
pip install woodelf_explainer
python -c "import os; os.chdir('..'); os.chdir('..'); import woodelf; print(woodelf); print(woodelf.__version__)"

# Add the version to git
X.Y.Z should be replaced with the new version
git tag vX.Y.Z
git push origin vX.Y.Z