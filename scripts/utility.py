
import sys
import tellurium
import roadrunner
import json


def convert_antimony_to_sbml(ant_fname:str, sbml_fname:str):
    '''load an antimony text file and save it to SBML xml'''
    try:
        rr = tellurium.loadAntimonyModel(ant_fname)
        rr.exportToSBML(sbml_fname)
    except:
        print('error: could not convert antimony to SBML')
        sys.exit(1)
    


def load_rr_model_from_sbml(fname:str) -> roadrunner.RoadRunner():
    '''load SBML model as a roadrunner object'''
    try:
        rr = roadrunner.RoadRunner(fname)
        return rr
    except:
        print('error: could not load roadrunner model from sbml file')
        sys.exit(1)


def get_rr_model_ODEs(rr_model) -> str():
    '''get ODEs describing roadrunner object'''
    ODEs = tellurium.getODEsFromModel(rr_model)
    return ODEs


def parse_config_file(config_fname:str) -> dict():
    '''opens a .json configuration file and returns a dictionary'''
    with open(config_fname) as json_file:
        data = json.load(json_file)
    return data

def save_config_file(json_data, file_name):
    '''save .json data with indentation for easier readability'''
    with open(file_name, 'w') as file:
        json.dump(json_data, file, indent=4)