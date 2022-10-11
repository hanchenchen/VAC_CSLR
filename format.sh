# remove unused imports
pycln . --exclude docker/
# sort import
isort -rc .
# format code
black . --exclude docker/
