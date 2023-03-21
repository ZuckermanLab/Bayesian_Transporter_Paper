import libsbml
import tellurium as te
import numpy as np


def validate_model(sbml_file):
    # Load the SBML file
    reader = libsbml.SBMLReader()
    document = reader.readSBMLFromFile(sbml_file)
    if document.getNumErrors() > 0:
        print("Errors occurred while reading the SBML file:")
        print(document.getErrorLog().toString())
        return False
    if document.checkConsistency() > 0:
        print("warnings occurred while reading the SBML file:")
        print(document.getErrorLog().toString())
    if document.checkInternalConsistency() > 0:
        print("internal consistency warnings occurred while reading the SBML file:")
        print(document.getErrorLog().toString())

    model = document.getModel()

    # Check that all species, reactions, and compartments have unique identifiers
    ids = set()
    for species in model.getListOfSpecies():
        id = species.getId()
        if id in ids:
            print(f"Duplicate identifier found for species {id}")
            return False
        ids.add(id)
    for reaction in model.getListOfReactions():
        id = reaction.getId()
        if id in ids:
            print(f"Duplicate identifier found for reaction {id}")
            return False
        ids.add(id)
    for compartment in model.getListOfCompartments():
        id = compartment.getId()
        if id in ids:
            print(f"Duplicate identifier found for compartment {id}")
            return False
        ids.add(id)

    # Ensure that all species have a positive initial concentration or amount
    for species in model.getListOfSpecies():
        if species.isSetInitialAmount() and species.getInitialAmount() < 0:
            print(f"Species {species.getId()} has non-positive initial amount")
            return False
        if species.isSetInitialConcentration() and species.getInitialConcentration() < 0:
            print(f"Species {species.getId()} has non-positive initial concentration")
            return False

    # Verify that all reaction stoichiometries are balanced
    for reaction in model.getListOfReactions():
        for species_reference in reaction.getListOfReactants():
            species = model.getSpecies(species_reference.getSpecies())
            if species is None:
                print(f"Reactant species {species_reference.getSpecies()} not found in model")
                return False
            if not species.isSetInitialAmount() and not species.isSetInitialConcentration():
                print(f"Reactant species {species.getId()} has no initial amount or concentration")
                return False
            if species_reference.getStoichiometry() * species.getInitialAmount() < 0:
                print(f"Reactant species {species.getId()} has non-positive initial amount")
                return False
        for species_reference in reaction.getListOfProducts():
            species = model.getSpecies(species_reference.getSpecies())
            if species is None:
                print(f"Product species {species_reference.getSpecies()} not found in model")
                return False
            if not species.isSetInitialAmount() and not species.isSetInitialConcentration():
                print(f"Product species {species.getId()} has no initial amount or concentration")
                return False
            if species_reference.getStoichiometry() * species.getInitialAmount() < 0:
                print(f"Product species {species.getId()} has non-positive initial amount")
                return False
            
        # Check rate laws 
        math = reaction.getKineticLaw().getMath()

        # Check if the rate law is defined
        if math is None:
            print(f"Rate law for reaction {reaction.getId()} is not defined")
            continue

        # Check if the rate law is valid
        try:
            rate_law = libsbml.formulaToString(math)
            print(rate_law)
        except:
            print(f"Invalid rate law for reaction {reaction.getId()}: {rate_law}")

    print('check online SBML validation tool for more in-depth model validation: https://sbml.org/validator_servlet/validate.jsp')



def validate_model_simulation(sbml_file):
    # load SBML model
    model = te.loadSBMLModel(sbml_file)

    # simulate model
    model.integrator.absolute_tolerance = 1e-22
    model.integrator.relative_tolerance = 1e-12
    sim = model.simulate()

    # check if any concentrations are negative
    assert(np.any((np.array(sim) >= 0))==True)


### validate SBML model
sbml_file = "data/antiporter_12D_model_3c_no_events.xml"
validate_model(sbml_file)
validate_model_simulation(sbml_file)

sbml_file = "data/antiporter_15D_model_3c_no_events.xml"
validate_model(sbml_file)
validate_model_simulation(sbml_file)
