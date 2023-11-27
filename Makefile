.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y forests || :
	@pip install -e .

##################### TESTS #####################


################### DATA SOURCES ACTIONS ################

