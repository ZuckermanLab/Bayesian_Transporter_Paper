##### 6 parameter transporter model sampling - August George - 2021 

import numpy as np
from emcee import PTSampler
import tellurium as te
import pandas as pd
import time
import pathlib
#import multiprocessing as mp
#import multiprocess as mp
import ray.util.multiprocessing as mp
import matplotlib.pyplot as plt
import corner
import os
#os.environ["OMP_NUM_THREADS"] = "1"







def initialize_model(p):
    ''' get initial model (from random walker parameter set)'''
    k_conf = 10 ** p[0]
    k_H_on = 10 ** p[1]
    k_S_on = 10 ** p[2]
    k_H_off = 10 ** p[3]
    k_S_off = 10 ** p[4]
    _ = 10**p[5]
    # note: p[-1] is sigma (not used in transporter model calculation)

    if model_id==1:
        ### default model
        antimony_string = f"""
            // Created by libAntimony v2.12.0
            model transporter_full()

            // Compartments and Species:
            compartment compartment_;
            species $H_out in compartment_, OF in compartment_, OF_Hb in compartment_;
            species IF_Hb in compartment_, S_in in compartment_, IF_Hb_Sb in compartment_;
            species H_in in compartment_, IF_Sb in compartment_, OF_Sb in compartment_;
            species $S_out in compartment_, IF in compartment_, OF_Hb_Sb in compartment_;

            // Reactions:
            rxn1: IF -> OF; compartment_*(rxn1_k1*IF - rxn1_k2*OF);
            rxn2: OF + $H_out -> OF_Hb; compartment_*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb);
            rxn3: OF_Sb -> OF + $S_out; compartment_*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out);
            rxn4: OF_Hb -> IF_Hb; compartment_*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb);
            rxn5: OF_Hb_Sb -> OF_Hb + $S_out; compartment_*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out);
            rxn6: IF_Sb -> OF_Sb; compartment_*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb);
            rxn7: OF_Sb + $H_out -> OF_Hb_Sb; compartment_*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb);
            rxn8: OF_Hb_Sb -> IF_Hb_Sb; compartment_*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb);
            rxn9: IF_Hb -> IF + H_in; compartment_*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in);
            rxn10: IF + S_in -> IF_Sb; compartment_*(rxn10_k1*IF*S_in - rxn10_k2*IF_Sb);
            rxn11: IF_Hb + S_in -> IF_Hb_Sb; compartment_*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb);
            rxn12: IF_Hb_Sb -> IF_Sb + H_in; compartment_*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in);

            // Events:
            E1: at (time >= 5.0): H_out = H_out_activation;
            E2: at (time >= 10.0): H_out = 1e-7;

            // Species initializations:
            H_out = 1e-07;
            H_out has substance_per_volume;

            OF = 2.833e-8;
            OF has substance_per_volume;

            OF_Hb = 2.833e-8;
            OF_Hb has substance_per_volume;

            IF_Hb = 2.833e-8;
            IF_Hb has substance_per_volume;

            S_in = 1e-3;
            S_in has substance_per_volume;

            IF_Hb_Sb = 2.833e-8;
            IF_Hb_Sb has substance_per_volume;

            H_in = 1e-7;
            H_in has substance_per_volume;

            IF_Sb = 2.125e-08;
            IF_Sb has substance_per_volume;

            OF_Sb = 2.125e-08;
            OF_Sb has substance_per_volume;

            S_out = 0.001;
            S_out has substance_per_volume;

            IF = 0;
            IF has substance_per_volume;

            OF_Hb_Sb = 0;
            OF_Hb_Sb has substance_per_volume;

            // Compartment initializations:
            compartment_ = 0.0001;
            compartment_ has volume;

            // Variable initializations:
            H_out_activation = 5e-8;
            k_conf = {k_conf};
            k_S_on = {k_S_on};
            k_S_off = {k_S_off};
            k_H_on = {k_H_on};
            k_H_off = {k_H_off};

            // Variable initializations:
            rxn1_k1 = 0;
            rxn1_k2 = 0;
            rxn2_k1 = k_H_on;
            rxn2_k2 = k_H_off;
            rxn3_k1 = k_S_off;
            rxn3_k2 = k_S_on;
            rxn4_k1 = k_conf;
            rxn4_k2 = k_conf;
            rxn5_k1 = 0;
            rxn5_k2 = 0;
            rxn6_k1 = k_conf;
            rxn6_k2 = k_conf;
            rxn7_k1 = 0;
            rxn7_k2 = 0;
            rxn8_k1 = 0;
            rxn8_k2 = 0;
            rxn9_k1 = 0;
            rxn9_k2 = 0;
            rxn10_k1 = 0;
            rxn10_k2 = 0;
            rxn11_k1 = k_S_on;
            rxn11_k2 = k_S_off;
            rxn12_k1 = k_H_off;
            rxn12_k2 = k_H_on;


            // Other declarations:
            const compartment_, rxn1_k1, rxn1_k2, rxn2_k1, rxn2_k2, rxn3_k1, rxn3_k2;
            const rxn4_k1, rxn4_k2, rxn5_k1, rxn5_k2, rxn6_k1, rxn6_k2, rxn7_k1, rxn7_k2;
            const rxn8_k1, rxn8_k2, rxn9_k1, rxn9_k2, rxn10_k1, rxn10_k2, rxn11_k1;
            const rxn11_k2, rxn12_k1, rxn12_k2, k_conf, k_S_on, k_H_on, k_S_off, k_H_off;
            # const k_off;

            // Unit definitions:
            unit substance_per_volume = mole / litre;
            unit volume = litre;
            unit length = metre;
            unit area = metre^2;
            unit time_unit = second;
            unit substance = mole;
            unit extent = mole;

            // Display Names:
            time_unit is "time";
            end
            """ 
    elif model_id==2:
        antimony_string = f"""
            // Created by libAntimony v2.12.0
            model transporter_full()

            // Compartments and Species:
            compartment compartment_;
            species $H_out in compartment_, OF in compartment_, OF_Hb in compartment_;
            species IF_Hb in compartment_, S_in in compartment_, IF_Hb_Sb in compartment_;
            species H_in in compartment_, IF_Sb in compartment_, OF_Sb in compartment_;
            species $S_out in compartment_, IF in compartment_, OF_Hb_Sb in compartment_;

            // Reactions:
            rxn1: IF -> OF; compartment_*(rxn1_k1*IF - rxn1_k2*OF);
            rxn2: OF + $H_out -> OF_Hb; compartment_*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb);
            rxn3: OF_Sb -> OF + $S_out; compartment_*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out);
            rxn4: OF_Hb -> IF_Hb; compartment_*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb);
            rxn5: OF_Hb_Sb -> OF_Hb + $S_out; compartment_*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out);
            rxn6: IF_Sb -> OF_Sb; compartment_*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb);
            rxn7: OF_Sb + $H_out -> OF_Hb_Sb; compartment_*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb);
            rxn8: OF_Hb_Sb -> IF_Hb_Sb; compartment_*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb);
            rxn9: IF_Hb -> IF + H_in; compartment_*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in);
            rxn10: IF + S_in -> IF_Sb; compartment_*(rxn10_k1*IF*S_in - rxn10_k2*IF_Sb);
            rxn11: IF_Hb + S_in -> IF_Hb_Sb; compartment_*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb);
            rxn12: IF_Hb_Sb -> IF_Sb + H_in; compartment_*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in);

            // Events:
            E1: at (time >= 5.0): H_out = H_out_activation;
            E2: at (time >= 10.0): H_out = 1e-7;

            // Species initializations:
            H_out = 1e-07;
            H_out has substance_per_volume;

            OF = 2.833e-8;
            OF has substance_per_volume;

            OF_Hb = 2.833e-8;
            OF_Hb has substance_per_volume;

            IF_Hb = 2.833e-8;
            IF_Hb has substance_per_volume;

            S_in = 1e-3;
            S_in has substance_per_volume;

            IF_Hb_Sb = 0;
            IF_Hb_Sb has substance_per_volume;

            H_in = 1e-7;
            H_in has substance_per_volume;

            IF_Sb = 2.125e-08;
            IF_Sb has substance_per_volume;

            OF_Sb = 2.125e-08;
            OF_Sb has substance_per_volume;

            S_out = 0.001;
            S_out has substance_per_volume;

            IF = 2.833e-8;
            IF has substance_per_volume;

            OF_Hb_Sb = 0;
            OF_Hb_Sb has substance_per_volume;

            // Compartment initializations:
            compartment_ = 0.0001;
            compartment_ has volume;

            // Variable initializations:
            H_out_activation = 5e-8;
            k_conf = {k_conf};
            k_S_on = {k_S_on};
            k_S_off = {k_S_off};
            k_H_on = {k_H_on};
            k_H_off = {k_H_off};

            // Variable initializations:
            rxn1_k1 = 0;
            rxn1_k2 = 0;
            rxn2_k1 = k_H_on;
            rxn2_k2 = k_H_off;
            rxn3_k1 = k_S_off;
            rxn3_k2 = k_S_on;
            rxn4_k1 = k_conf;
            rxn4_k2 = k_conf;
            rxn5_k1 = 0;
            rxn5_k2 = 0;
            rxn6_k1 = k_conf;
            rxn6_k2 = k_conf;
            rxn7_k1 = 0;
            rxn7_k2 = 0;
            rxn8_k1 = 0;
            rxn8_k2 = 0;
            rxn9_k1 = k_H_off;
            rxn9_k2 = k_H_on;
            rxn10_k1 = k_S_on;
            rxn10_k2 = k_S_off;
            rxn11_k1 = 0;
            rxn11_k2 = 0;
            rxn12_k1 = 0;
            rxn12_k2 = 0;


            // Other declarations:
            const compartment_, rxn1_k1, rxn1_k2, rxn2_k1, rxn2_k2, rxn3_k1, rxn3_k2;
            const rxn4_k1, rxn4_k2, rxn5_k1, rxn5_k2, rxn6_k1, rxn6_k2, rxn7_k1, rxn7_k2;
            const rxn8_k1, rxn8_k2, rxn9_k1, rxn9_k2, rxn10_k1, rxn10_k2, rxn11_k1;
            const rxn11_k2, rxn12_k1, rxn12_k2, k_conf, k_S_on, k_H_on, k_S_off, k_H_off;
            # const k_off;

            // Unit definitions:
            unit substance_per_volume = mole / litre;
            unit volume = litre;
            unit length = metre;
            unit area = metre^2;
            unit time_unit = second;
            unit substance = mole;
            unit extent = mole;

            // Display Names:
            time_unit is "time";
            end
            """ 
    elif model_id==3:
        antimony_string = f"""
            // Created by libAntimony v2.12.0
            model transporter_full()

            // Compartments and Species:
            compartment compartment_;
            species $H_out in compartment_, OF in compartment_, OF_Hb in compartment_;
            species IF_Hb in compartment_, S_in in compartment_, IF_Hb_Sb in compartment_;
            species H_in in compartment_, IF_Sb in compartment_, OF_Sb in compartment_;
            species $S_out in compartment_, IF in compartment_, OF_Hb_Sb in compartment_;

            // Reactions:
            rxn1: IF -> OF; compartment_*(rxn1_k1*IF - rxn1_k2*OF);
            rxn2: OF + $H_out -> OF_Hb; compartment_*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb);
            rxn3: OF_Sb -> OF + $S_out; compartment_*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out);
            rxn4: OF_Hb -> IF_Hb; compartment_*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb);
            rxn5: OF_Hb_Sb -> OF_Hb + $S_out; compartment_*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out);
            rxn6: IF_Sb -> OF_Sb; compartment_*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb);
            rxn7: OF_Sb + $H_out -> OF_Hb_Sb; compartment_*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb);
            rxn8: OF_Hb_Sb -> IF_Hb_Sb; compartment_*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb);
            rxn9: IF_Hb -> IF + H_in; compartment_*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in);
            rxn10: IF + S_in -> IF_Sb; compartment_*(rxn10_k1*IF*S_in - rxn10_k2*IF_Sb);
            rxn11: IF_Hb + S_in -> IF_Hb_Sb; compartment_*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb);
            rxn12: IF_Hb_Sb -> IF_Sb + H_in; compartment_*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in);

            // Events:
            E1: at (time >= 5.0): H_out = H_out_activation;
            E2: at (time >= 10.0): H_out = 1e-7;

            // Species initializations:
            H_out = 1e-07;
            H_out has substance_per_volume;

            OF = 0;
            OF has substance_per_volume;

            OF_Hb = 2.833e-8;
            OF_Hb has substance_per_volume;

            IF_Hb = 2.833e-8;
            IF_Hb has substance_per_volume;

            S_in = 1e-3;
            S_in has substance_per_volume;

            IF_Hb_Sb = 0;
            IF_Hb_Sb has substance_per_volume;

            H_in = 1e-7;
            H_in has substance_per_volume;

            IF_Sb = 2.125e-08;
            IF_Sb has substance_per_volume;

            OF_Sb = 2.125e-08;
            OF_Sb has substance_per_volume;

            S_out = 0.001;
            S_out has substance_per_volume;

            IF = 2.833e-8;
            IF has substance_per_volume;

            OF_Hb_Sb = 2.833e-8;
            OF_Hb_Sb has substance_per_volume;

            // Compartment initializations:
            compartment_ = 0.0001;
            compartment_ has volume;

            // Variable initializations:
            H_out_activation = 5e-8;
            k_conf = {k_conf};
            k_S_on = {k_S_on};
            k_S_off = {k_S_off};
            k_H_on = {k_H_on};
            k_H_off = {k_H_off};

            // Variable initializations:
            rxn1_k1 = 0;
            rxn1_k2 = 0;
            rxn2_k1 = 0;
            rxn2_k2 = 0;
            rxn3_k1 = 0;
            rxn3_k2 = 0;
            rxn4_k1 = k_conf;
            rxn4_k2 = k_conf;
            rxn5_k1 = k_S_off;
            rxn5_k2 = k_S_on;
            rxn6_k1 = k_conf;
            rxn6_k2 = k_conf;
            rxn7_k1 = k_H_on;
            rxn7_k2 = k_H_off;
            rxn8_k1 = 0;
            rxn8_k2 = 0;
            rxn9_k1 = k_H_off;
            rxn9_k2 = k_H_on;
            rxn10_k1 = k_S_on;
            rxn10_k2 = k_S_off;
            rxn11_k1 = 0;
            rxn11_k2 = 0;
            rxn12_k1 = 0;
            rxn12_k2 = 0;


            // Other declarations:
            const compartment_, rxn1_k1, rxn1_k2, rxn2_k1, rxn2_k2, rxn3_k1, rxn3_k2;
            const rxn4_k1, rxn4_k2, rxn5_k1, rxn5_k2, rxn6_k1, rxn6_k2, rxn7_k1, rxn7_k2;
            const rxn8_k1, rxn8_k2, rxn9_k1, rxn9_k2, rxn10_k1, rxn10_k2, rxn11_k1;
            const rxn11_k2, rxn12_k1, rxn12_k2, k_conf, k_S_on, k_H_on, k_S_off, k_H_off;
            # const k_off;

            // Unit definitions:
            unit substance_per_volume = mole / litre;
            unit volume = litre;
            unit length = metre;
            unit area = metre^2;
            unit time_unit = second;
            unit substance = mole;
            unit extent = mole;

            // Display Names:
            time_unit is "time";
            end
            """ 
    elif model_id==4:
        antimony_string = f"""
            // Created by libAntimony v2.12.0
            model transporter_full()

            // Compartments and Species:
            compartment compartment_;
            species $H_out in compartment_, OF in compartment_, OF_Hb in compartment_;
            species IF_Hb in compartment_, S_in in compartment_, IF_Hb_Sb in compartment_;
            species H_in in compartment_, IF_Sb in compartment_, OF_Sb in compartment_;
            species $S_out in compartment_, IF in compartment_, OF_Hb_Sb in compartment_;

            // Reactions:
            rxn1: IF -> OF; compartment_*(rxn1_k1*IF - rxn1_k2*OF);
            rxn2: OF + $H_out -> OF_Hb; compartment_*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb);
            rxn3: OF_Sb -> OF + $S_out; compartment_*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out);
            rxn4: OF_Hb -> IF_Hb; compartment_*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb);
            rxn5: OF_Hb_Sb -> OF_Hb + $S_out; compartment_*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out);
            rxn6: IF_Sb -> OF_Sb; compartment_*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb);
            rxn7: OF_Sb + $H_out -> OF_Hb_Sb; compartment_*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb);
            rxn8: OF_Hb_Sb -> IF_Hb_Sb; compartment_*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb);
            rxn9: IF_Hb -> IF + H_in; compartment_*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in);
            rxn10: IF + S_in -> IF_Sb; compartment_*(rxn10_k1*IF*S_in - rxn10_k2*IF_Sb);
            rxn11: IF_Hb + S_in -> IF_Hb_Sb; compartment_*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb);
            rxn12: IF_Hb_Sb -> IF_Sb + H_in; compartment_*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in);

            // Events:
            E1: at (time >= 5.0): H_out = H_out_activation;
            E2: at (time >= 10.0): H_out = 1e-7;

            // Species initializations:
            H_out = 1e-07;
            H_out has substance_per_volume;

            OF = 0;
            OF has substance_per_volume;

            OF_Hb = 2.833e-8;
            OF_Hb has substance_per_volume;

            IF_Hb = 2.833e-8;
            IF_Hb has substance_per_volume;

            S_in = 1e-3;
            S_in has substance_per_volume;

            IF_Hb_Sb = 2.833e-8;
            IF_Hb_Sb has substance_per_volume;

            H_in = 1e-7;
            H_in has substance_per_volume;

            IF_Sb = 2.125e-08;
            IF_Sb has substance_per_volume;

            OF_Sb = 2.125e-08;
            OF_Sb has substance_per_volume;

            S_out = 0.001;
            S_out has substance_per_volume;

            IF = 0;
            IF has substance_per_volume;

            OF_Hb_Sb = 2.833e-8;
            OF_Hb_Sb has substance_per_volume;

            // Compartment initializations:
            compartment_ = 0.0001;
            compartment_ has volume;

            // Variable initializations:
            H_out_activation = 5e-8;
            k_conf = {k_conf};
            k_S_on = {k_S_on};
            k_S_off = {k_S_off};
            k_H_on = {k_H_on};
            k_H_off = {k_H_off};

            // Variable initializations:
            rxn1_k1 = 0;
            rxn1_k2 = 0;
            rxn2_k1 = 0;
            rxn2_k2 = 0;
            rxn3_k1 = 0;
            rxn3_k2 = 0;
            rxn4_k1 = k_conf;
            rxn4_k2 = k_conf;
            rxn5_k1 = k_S_off;
            rxn5_k2 = k_S_on;
            rxn6_k1 = k_conf;
            rxn6_k2 = k_conf;
            rxn7_k1 = k_H_on;
            rxn7_k2 = k_H_off;
            rxn8_k1 = 0;
            rxn8_k2 = 0;
            rxn9_k1 = 0;
            rxn9_k2 = 0;
            rxn10_k1 = 0;
            rxn10_k2 = 0;
            rxn11_k1 = k_S_on;
            rxn11_k2 = k_S_off;
            rxn12_k1 = k_H_off;
            rxn12_k2 = k_H_on;

            // Other declarations:
            const compartment_, rxn1_k1, rxn1_k2, rxn2_k1, rxn2_k2, rxn3_k1, rxn3_k2;
            const rxn4_k1, rxn4_k2, rxn5_k1, rxn5_k2, rxn6_k1, rxn6_k2, rxn7_k1, rxn7_k2;
            const rxn8_k1, rxn8_k2, rxn9_k1, rxn9_k2, rxn10_k1, rxn10_k2, rxn11_k1;
            const rxn11_k2, rxn12_k1, rxn12_k2, k_conf, k_S_on, k_H_on, k_S_off, k_H_off;
            # const k_off;

            // Unit definitions:
            unit substance_per_volume = mole / litre;
            unit volume = litre;
            unit length = metre;
            unit area = metre^2;
            unit time_unit = second;
            unit substance = mole;
            unit extent = mole;

            // Display Names:
            time_unit is "time";
            end
            """ 
        

    z = te.loada(antimony_string)
    return z


def get_y_pred(p,z):
    '''generates flux trace based on a set of parameters, p, and a Tellurium transporter model, z'''

    _ = 10**p[5]
    # reset z to initial
    z.resetToOrigin()

    # update theta
    k_conf_tmp = 10**p[0]
    k_H_on_tmp = 10**p[1]
    k_S_on_tmp = 10**p[2]
    k_H_off_tmp = 10**p[3]
    k_S_off_tmp = 10**p[4]
    z.k_conf = k_conf_tmp
    z.k_H_on = k_H_on_tmp
    z.k_S_on = k_S_on_tmp
    z.k_H_off= k_H_off_tmp
    z.k_S_off= k_S_off_tmp
    z.rxn2_k1 = k_H_on_tmp
    z.rxn2_k2 = k_H_off_tmp
    z.rxn3_k1 = k_S_off_tmp
    z.rxn3_k2 = k_S_on_tmp
    z.rxn4_k1 = k_conf_tmp
    z.rxn4_k2 = k_conf_tmp
    z.rxn6_k1 = k_conf_tmp
    z.rxn6_k2 = k_conf_tmp
    z.rxn11_k1 = k_S_on_tmp
    z.rxn11_k2 = k_S_off_tmp
    z.rxn12_k1 = k_H_off_tmp
    z.rxn12_k2 = k_H_on_tmp


    # set tolerances for simulations
    z.integrator.absolute_tolerance = 1e-19
    z.integrator.relative_tolerance = 1e-17

    n_stage = 3  # number of stages: equilibration, activation, reversal
    t_stage = 5  # time length for each stage (in sec) to allow for equilibration
    n_iter_stage = 5e3  # how many how many ODE solver iterations per stage
    t_res = 0.04  # time resolution (sec)
    n_samples_stage = int(t_stage / t_res)  # how many data points per stage

    t_0 = 0
    t_f = int(np.floor(n_stage * t_stage))
    n_iter = int(np.floor(n_iter_stage * n_stage))
    idx_s2 = int(np.floor(n_iter_stage))
    step_size = int(np.floor(n_iter_stage / n_samples_stage))

    # try:        
    #     D = pd.DataFrame(z.simulate(t_0, t_f, n_iter, selections=['time', 'rxn9', 'rxn12']),
    #                     columns=['time', 'rxn9', 'rxn12'])
    #     y_calc = pd.DataFrame(D['rxn9'] + D['rxn12'], columns=['y_calc'])  # rxn 9 + rxn 12
    #     #y_pred = y_calc.iloc[idx_s2+4:2*idx_s2:step_size]
    #     #y_pred = y_calc.iloc[idx_s2::step_size]
    #     s2 = y_calc.iloc[idx_s2+4:2*idx_s2:step_size] 
    #     s3 = y_calc.iloc[2*idx_s2+4::step_size]
    #     y_pred = pd.concat([s2,s3]) 
    # except:
    #     print('error in y_pred calculations')
    #     y_error_list = [0 * i for i in range(125)]  # datapoints?
    #     y_pred = pd.DataFrame(y_error_list, columns='y_calc')
    #     with open(f'emcee_transporter_error.txt', 'a') as f:
    #         f.write(f'\nERROR IN Y_PRED CALCULATIONS!\nSetting y_pred={y_error_list}\n')
    # return y_pred['y_calc'].values
            
    D = z.simulate(t_0, t_f, n_iter, selections=['time', 'rxn9', 'rxn12'])
    y_calc = D['rxn9']+D['rxn12']
    #s1 = y_calc[:idx_s2+4:step_size]
    s2 = y_calc[idx_s2+4:2*idx_s2:step_size]
    s3 = y_calc[2*idx_s2+4::step_size]

    y_pred = np.hstack([s2,s3])
    return y_pred


def get_y_pred_2ph(p,z):
    '''generates flux trace based on a set of parameters, p, and a Tellurium transporter model, z'''

    _ = 10**p[5]
    # reset z to initial
    z.resetToOrigin()

    #update pH
    z.H_out_activation = 5e-8
    # update theta
    k_conf_tmp = 10**p[0]
    k_H_on_tmp = 10**p[1]
    k_S_on_tmp = 10**p[2]
    k_H_off_tmp = 10**p[3]
    k_S_off_tmp = 10**p[4]

    # update rate constants
    z.k_conf = k_conf_tmp
    z.k_H_on = k_H_on_tmp
    z.k_S_on = k_S_on_tmp
    z.k_H_off= k_H_off_tmp
    z.k_S_off= k_S_off_tmp

    # model dependant
    if model_id==1:
        # model 1
        z.rxn2_k1 = k_H_on_tmp
        z.rxn2_k2 = k_H_off_tmp
        z.rxn3_k1 = k_S_off_tmp
        z.rxn3_k2 = k_S_on_tmp
        z.rxn4_k1 = k_conf_tmp
        z.rxn4_k2 = k_conf_tmp
        z.rxn6_k1 = k_conf_tmp
        z.rxn6_k2 = k_conf_tmp
        z.rxn11_k1 = k_S_on_tmp
        z.rxn11_k2 = k_S_off_tmp
        z.rxn12_k1 = k_H_off_tmp
        z.rxn12_k2 = k_H_on_tmp
    elif model_id==2:
        # model 2
        z.rxn2_k1 = k_H_on_tmp
        z.rxn2_k2 = k_H_off_tmp
        z.rxn3_k1 = k_S_off_tmp
        z.rxn3_k2 = k_S_on_tmp
        z.rxn4_k1 = k_conf_tmp
        z.rxn4_k2 = k_conf_tmp
        z.rxn6_k1 = k_conf_tmp
        z.rxn6_k2 = k_conf_tmp
        z.rxn9_k1 = k_H_off_tmp
        z.rxn9_k2 = k_H_on_tmp
        z.rxn10_k1 = k_S_on_tmp
        z.rxn10_k2 = k_S_off_tmp
    elif model_id==3:
        # model 3
        z.rxn4_k1 = k_conf_tmp
        z.rxn4_k2 = k_conf_tmp
        z.rxn5_k1 = k_S_off_tmp;
        z.rxn5_k2 = k_S_on_tmp;
        z.rxn6_k1 = k_conf_tmp
        z.rxn6_k2 = k_conf_tmp
        z.rxn7_k1 = k_H_on_tmp
        z.rxn7_k2 = k_H_off_tmp
        z.rxn9_k1 = k_H_off_tmp
        z.rxn9_k2 = k_H_on_tmp
        z.rxn10_k1 = k_S_on_tmp
        z.rxn10_k2 = k_S_off_tmp
    elif model_id==4:
        # model 4
        z.rxn4_k1 = k_conf_tmp
        z.rxn4_k2 = k_conf_tmp
        z.rxn5_k1 = k_S_off_tmp;
        z.rxn5_k2 = k_S_on_tmp;
        z.rxn6_k1 = k_conf_tmp
        z.rxn6_k2 = k_conf_tmp
        z.rxn7_k1 = k_H_on_tmp
        z.rxn7_k2 = k_H_off_tmp
        z.rxn11_k1 = k_S_on_tmp
        z.rxn11_k2 = k_S_off_tmp
        z.rxn12_k1 = k_H_off_tmp
        z.rxn12_k2 = k_H_on_tmp

    

    # set tolerances for simulations
    z.integrator.absolute_tolerance = 1e-19
    z.integrator.relative_tolerance = 1e-17

    n_stage = 3  # number of stages: equilibration, activation, reversal
    t_stage = 5  # time length for each stage (in sec) to allow for equilibration
    n_iter_stage = 5e3  # how many how many ODE solver iterations per stage
    t_res = 0.04  # time resolution (sec)
    n_samples_stage = int(t_stage / t_res)  # how many data points per stage

    t_0 = 0
    t_f = int(np.floor(n_stage * t_stage))
    n_iter = int(np.floor(n_iter_stage * n_stage))
    idx_s2 = int(np.floor(n_iter_stage))
    step_size = int(np.floor(n_iter_stage / n_samples_stage))

        
    D = z.simulate(t_0, t_f, n_iter, selections=['time', 'rxn9', 'rxn12'])
    y_calc = D['rxn9']+D['rxn12']
    #s1 = y_calc[:idx_s2+4:step_size]
    s2 = y_calc[idx_s2+4:2*idx_s2:step_size]
    s3 = y_calc[2*idx_s2+4::step_size]

    y_pred_1 = np.hstack([s2,s3])

    # reset z to initial
    z.resetToOrigin()

    #update pH
    z.H_out_activation = 5e-7

    # update theta
    k_conf_tmp = 10**p[0]
    k_H_on_tmp = 10**p[1]
    k_S_on_tmp = 10**p[2]
    k_H_off_tmp = 10**p[3]
    k_S_off_tmp = 10**p[4]

    # assign rate constants
    z.k_conf = k_conf_tmp
    z.k_H_on = k_H_on_tmp
    z.k_S_on = k_S_on_tmp
    z.k_H_off= k_H_off_tmp
    z.k_S_off= k_S_off_tmp

    # model dependant
    if model_id==1:
        # model 1
        z.rxn2_k1 = k_H_on_tmp
        z.rxn2_k2 = k_H_off_tmp
        z.rxn3_k1 = k_S_off_tmp
        z.rxn3_k2 = k_S_on_tmp
        z.rxn4_k1 = k_conf_tmp
        z.rxn4_k2 = k_conf_tmp
        z.rxn6_k1 = k_conf_tmp
        z.rxn6_k2 = k_conf_tmp
        z.rxn11_k1 = k_S_on_tmp
        z.rxn11_k2 = k_S_off_tmp
        z.rxn12_k1 = k_H_off_tmp
        z.rxn12_k2 = k_H_on_tmp
    elif model_id==2:
        # model 2
        z.rxn2_k1 = k_H_on_tmp
        z.rxn2_k2 = k_H_off_tmp
        z.rxn3_k1 = k_S_off_tmp
        z.rxn3_k2 = k_S_on_tmp
        z.rxn4_k1 = k_conf_tmp
        z.rxn4_k2 = k_conf_tmp
        z.rxn6_k1 = k_conf_tmp
        z.rxn6_k2 = k_conf_tmp
        z.rxn9_k1 = k_H_off_tmp
        z.rxn9_k2 = k_H_on_tmp
        z.rxn10_k1 = k_S_on_tmp
        z.rxn10_k2 = k_S_off_tmp
    elif model_id==3:
        # model 3
        z.rxn4_k1 = k_conf_tmp
        z.rxn4_k2 = k_conf_tmp
        z.rxn5_k1 = k_S_off_tmp
        z.rxn5_k2 = k_S_on_tmp
        z.rxn6_k1 = k_conf_tmp
        z.rxn6_k2 = k_conf_tmp
        z.rxn7_k1 = k_H_on_tmp
        z.rxn7_k2 = k_H_off_tmp
        z.rxn9_k1 = k_H_off_tmp
        z.rxn9_k2 = k_H_on_tmp
        z.rxn10_k1 = k_S_on_tmp
        z.rxn10_k2 = k_S_off_tmp
    elif model_id==4:
        # model 4
        z.rxn4_k1 = k_conf_tmp
        z.rxn4_k2 = k_conf_tmp
        z.rxn5_k1 = k_S_off_tmp;
        z.rxn5_k2 = k_S_on_tmp;
        z.rxn6_k1 = k_conf_tmp
        z.rxn6_k2 = k_conf_tmp
        z.rxn7_k1 = k_H_on_tmp
        z.rxn7_k2 = k_H_off_tmp
        z.rxn11_k1 = k_S_on_tmp
        z.rxn11_k2 = k_S_off_tmp
        z.rxn12_k1 = k_H_off_tmp
        z.rxn12_k2 = k_H_on_tmp

    # set tolerances for simulations
    z.integrator.absolute_tolerance = 1e-19
    z.integrator.relative_tolerance = 1e-17

    n_stage = 3  # number of stages: equilibration, activation, reversal
    t_stage = 5  # time length for each stage (in sec) to allow for equilibration
    n_iter_stage = 5e3  # how many how many ODE solver iterations per stage
    t_res = 0.04  # time resolution (sec)
    n_samples_stage = int(t_stage / t_res)  # how many data points per stage

    t_0 = 0
    t_f = int(np.floor(n_stage * t_stage))
    n_iter = int(np.floor(n_iter_stage * n_stage))
    idx_s2 = int(np.floor(n_iter_stage))
    step_size = int(np.floor(n_iter_stage / n_samples_stage))

        
    D2 = z.simulate(t_0, t_f, n_iter, selections=['time', 'rxn9', 'rxn12'])
    y_calc2 = D2['rxn9']+D2['rxn12']

    #s1_2 = y_calc2[:idx_s2+4:step_size]
    s2_2 = y_calc2[idx_s2+4:2*idx_s2:step_size]
    s3_2 = y_calc2[2*idx_s2+4::step_size]
    y_pred_2 = np.hstack([s2_2,s3_2])
    y_pred = np.hstack([y_pred_1, y_pred_2])

    return y_pred



def log_likelihood(theta, y_obs, model):
    '''log of Guassian likelihood distribution'''
    #print(theta)
    curr_sigma = 10**theta[5]
    y_pred = get_y_pred_2ph(theta, model)

    # calculate normal log likelihood
    logl = -len(y_obs) * np.log(np.sqrt(2.0 * np.pi) * curr_sigma)
    logl += -np.sum((y_obs - y_pred) ** 2.0) / (2.0 * curr_sigma ** 2.0) 
    return logl


def log_prior(theta):
    '''log of uniform prior distribution'''

    k_conf = theta[0]
    k_H_on = theta[1]
    k_S_on = theta[2]
    k_H_off = theta[3]
    k_S_off = theta[4]
    sigma = theta[5]

    # if prior is between boundary --> log(prior) = 0 (uninformitive prior)
    if np.log10(5e-15)<sigma<np.log10(5e-12) and -1<k_conf<5 and 7<k_H_on<13 and 4<k_S_on<10 \
        and 5<k_H_off<11 and 5<k_S_off<11:
        return 0  
    else:
        return -np.inf


def log_probability(theta, y_obs, model):
    '''log of estimated posterior probability'''
    logp = log_prior(theta)
    if not np.isfinite(logp):
        return -np.inf  # ~zero probability
    return logp + log_likelihood(theta, y_obs, model)  # log posterior ~ log likelihood + log prior

# global parameters
model_id=1
        
if __name__ == '__main__':

    
    #model_id=1
    assert(model_id in [1,2,3,4])
    print(f'using model {model_id}')

    ##### run 
        # global scope - fix later
    ##### input data #####
    datafile = 'emcee_transporter_data_2_2ph.csv'
    y_obs = np.loadtxt(f'{datafile}', delimiter=',', skiprows=1, usecols=1).tolist()  # load data from file

    ##### initial model #####
    seed = 1234
    np.random.seed(seed)
    sigma_init = np.random.uniform(np.log10(5e-15),np.log10(5e-12))
    k_conf_init = np.random.uniform(-1,5)
    k_H_on_init = np.random.uniform(7,13)
    k_S_on_init = np.random.uniform(4,10)
    k_H_off_init = np.random.uniform(0,6)
    k_S_off_init = np.random.uniform(0,6)
    p_init = [k_conf_init,k_H_on_init,k_S_on_init,k_H_off_init,k_S_off_init,sigma_init]
    model = initialize_model(p_init)
    #####

    ##### reference model settings #####   
    labels = ["k_conf", "k_H_on", "k_S_on", "k_H_off", "k_S_off", "sigma"]  
    p_labels = [
        'log10(k_conf)',
        'log10(k_H_on)',
        'log10(k_S_on)',
        'log10(k_H_off)',
        'log10(k_S_off)', 
        'log10(sigma)',
        ]  # parameter names (log10)
    theta_true = [1e2, 1e10, 1e7, 1e3, 1e3, 1e-13]  # theta and labels must be in same order
    print('debug: parallelization bug')
    theta_true_log = np.log10(theta_true)
    theta_ref = theta_true_log
    n_dim = len(theta_true)
    #####
  
    ##### sampling settings #####
    n_replicas = 1  # repeats using different starting points 
    n_walkers = 12 # at least 3x the number of parameters
    n_steps = int(1e5)  # at least 50x the autocorrelation time
    n_burn = int(0.1*n_steps)
    n_temps = 8
    parallel=False
    batch_sampling=False
    #####

    ##### output settings #####
    time_str = time.strftime("%Y%m%d_%H%M%S") 
    time_0 = time.time()
    new_dir = pathlib.Path(pathlib.Path.cwd(), f'{time_str}_emcee_transporter')
    new_dir.mkdir(parents=True, exist_ok=True)

    y_ref = get_y_pred_2ph(theta_ref,model)
    assert(len(y_obs)==len(y_ref))  # make sure dataset and predicted y have same length
    
    # # for plotting
    # plt.plot(y_obs ,'o', markersize=3, label='synthetic data')
    # plt.plot(y_ref, '--', label=f'transporter model {model_id}', alpha=0.7)
    # plt.ylabel('y')
    # plt.xlabel('t')
    # plt.legend()
    # plt.title('flux trace')
    # plt.savefig('y_ref.png')
    # exit()


    with open(new_dir/f'{time_str}_emcee_transporter_log.txt', 'a') as f:
        f.write(f'{time_str}_emcee_transporter_log.txt\n\n')
        f.write(f'timestamp:{time_str}\n')
        f.write(f'using model id: {model_id}\n')
        f.write(f'n replicas:{n_replicas}\nn walkers:{n_walkers}\nn steps/walker:{n_steps}\n' )
        f.write(f'n temps:{n_temps}\nn burn:{n_burn}\nseed:{seed}\n' )
        f.write(f'parallel:{parallel}\nbatch sampling:{batch_sampling}\n')
        f.write(f'datafile:{datafile}\nparameters:{labels}\nparameter reference values:{theta_true}\n')

    if parallel:
        with mp.Pool() as pool:
            ##### parallel tempering sampling
            sampler=PTSampler(n_temps, n_walkers, n_dim, log_likelihood, log_prior, loglargs=[y_obs, model], pool=pool )
            # random starts from uniform priors
            p0_list = []
            for j in range(n_temps):
                pos_list = []
                for i in range(n_walkers):
                    sigma_i = np.random.uniform(np.log10(5e-15),np.log10(5e-12))
                    k_conf_i = np.random.uniform(-1,5)
                    k_H_on_i = np.random.uniform(7,13)
                    k_S_on_i = np.random.uniform(4,10)
                    k_H_off_i = np.random.uniform(0,6)
                    k_S_off_i = np.random.uniform(0,6)
                    pos_list.append([k_conf_i,k_H_on_i,k_S_on_i,k_H_off_i,k_S_off_i,sigma_i])
                p0_list.append(pos_list)
            p0 = np.asarray(p0_list)
            assert(np.shape(p0) == (n_temps,n_walkers,n_dim))

            i=0
            for p, lnprob, lnlike in sampler.sample(p0,iterations=int(n_steps+n_burn)):
                if i%100 == 0:
                    print(f'{i+1}/{n_steps}')
                i+=1
                pass
            assert sampler.chain.shape == (n_temps, n_walkers, int(n_steps+n_burn), n_dim)
    else:
        sampler=PTSampler(n_temps, n_walkers, n_dim, log_likelihood, log_prior, loglargs=[y_obs, model])
        # random starts from uniform priors
        p0_list = []
        for j in range(n_temps):
            pos_list = []
            for i in range(n_walkers):
                sigma_i = np.random.uniform(np.log10(5e-15),np.log10(5e-12))
                k_conf_i = np.random.uniform(-1,5)
                k_H_on_i = np.random.uniform(7,13)
                k_S_on_i = np.random.uniform(4,10)
                k_H_off_i = np.random.uniform(0,6)
                k_S_off_i = np.random.uniform(0,6)
                pos_list.append([k_conf_i,k_H_on_i,k_S_on_i,k_H_off_i,k_S_off_i,sigma_i])
            p0_list.append(pos_list)
        p0 = np.asarray(p0_list)
        assert(np.shape(p0) == (n_temps,n_walkers,n_dim))

        i=0
        for p, lnprob, lnlike in sampler.sample(p0, iterations=n_burn):
            if i%100 == 0:
                print(f'{i+1}/{n_burn}')
            i+=1
        sampler.reset()

        i=0
        for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                                lnlike0=lnlike,
                                                iterations=n_steps):
            if i%100 == 0:
                print(f'{i+1}/{n_steps}')
            i+=1
            pass
        assert sampler.chain.shape == (n_temps, n_walkers, n_steps, n_dim)

    with open(new_dir/f'{time_str}_emcee_transporter_log.txt', 'a') as f:
        f.write(f'sampling wall clock (s): {time.time()-time_0}\n\n')    


    ### data output    
    flat_lnlike = sampler.lnlikelihood[0,:,:].reshape((-1))
    flat_samples = sampler.flatchain[0,:,:]
    df_samples = pd.DataFrame(flat_samples, columns=p_labels)
    df_samples['logl'] = flat_lnlike
    df_samples.to_csv(new_dir/'pt_output_burn_removed.csv')

    ### basic analysis
    with open(new_dir/f'{time_str}_emcee_transporter_log.txt', 'a') as f:
        f.write(f'log-likelihood max: {flat_lnlike.max()}\n')
        f.write(f'log-likelihood min: {flat_lnlike.min()}\n')
        f.write(f'log-likelihood mean: {flat_lnlike.mean()}\n')
        f.write(f'log-likelihood stdev.: {np.std(flat_lnlike)}\n')             
  

    ### data visualization
    n_bins = 100
    n_p = len(p_init)
    bounds = [
        (-1,5),
        (7,13), 
        (4,10),
        (0,6),
        (0,6),
        (np.log10(5e-14),np.log10(5e-13)),
    ] # parameter boundaries (min,max)
    ref = [2,10,7,3,3,-13]  # reference values (leave empty if none)
    flat_samples_T = np.transpose(flat_samples)

    # corner plots
    fig = corner.corner(
                    flat_samples, bins=n_bins, labels=p_labels)
    plt.suptitle('low T')
    plt.savefig(new_dir/'pair_low_t.png')


    flat_samples2 = sampler.flatchain[-1,:,:]  # high T
    fig = corner.corner(
                    flat_samples2, bins=n_bins, labels=p_labels)
    plt.suptitle('high T')
    plt.savefig(new_dir/'pair_high_t.png')



    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(15,10))
    plt.suptitle(f'MCMC sampling results')
    # 1d posterior and mcmc traces plot
    for i in range(n_p):
        p_tmp = flat_samples_T[i]

        ### histograms
        hist_tmp, edges_tmp = np.histogram(p_tmp, bins=n_bins, range=(bounds[i][0], bounds[i][1]), density=True)  # make histogram (density)
        ax[i,0].bar(edges_tmp[:-1], hist_tmp, width=np.diff(edges_tmp), align='edge',alpha = 0.5)  # plot histogram
        if ref:  # add ref line
            ax[i,0].vlines(ref[i], 0, 2, ls='--', color='black', alpha=0.5)
        ax[i,0].set_title(p_labels[i])
        ax[i,0].set_ylabel('density')
        ax[i,0].set_xlabel('x')
        ax[i,0].set_xlim(ref[i]-3,ref[i]+3)

        ### mcmc chains
        for m in range(n_walkers):
            p_tmp2 = p_tmp[m*n_steps:(m+1)*n_steps:]
            ax[i,1].plot(p_tmp2, alpha=0.25)
        ax[i,1].set_ylim(ref[i]-3, ref[i]+3)
        if ref:
            ax[i,1].axhline(ref[i], color='k', ls='--', alpha=0.6)
        ax[i,1].set_title(f'{p_labels[i]}')
        ax[i,1].set_ylabel('y')
        ax[i,1].set_xlabel('MCMC n')
        ax[i,1].set_ylim(ref[i]-3,ref[i]+3)
    plt.tight_layout()
    plt.savefig(new_dir/'mcmc_sampling.png')


    # y pred vs y obs plots
    fig = plt.figure(figsize=(15,10))
    n_traces = 100

    plt.title(f'flux trace: {n_traces} predicted vs observed')
    
    for i in range(n_traces):
        p_tmp = flat_samples[np.random.randint(0,len(flat_samples),1)][0]
        y_pred_i = get_y_pred_2ph(p_tmp,model)
        if i == n_traces-1:
            plt.plot(y_pred_i, color='k', alpha=0.1, label='predicted')
        else:
            plt.plot(y_pred_i, color='k', alpha=0.1)
    plt.plot(y_obs, 'o', color='red', markersize=4, label='synthetic data', alpha=0.6)
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('t')
    plt.tight_layout()
    plt.savefig(new_dir/'y_pred_trace.png')
    





 
