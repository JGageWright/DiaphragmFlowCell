import numpy as np
import pandas as pd
import pyequion
import math

## Setup log
import logging.config, logging
logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True})
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s - %(message)s')
log  = logging.getLogger(__name__)
log.propagate = False

def singleCondition(ra: float, rc: float, electrolyte: str, j: float, previous_condition: tuple=None, w: float=0.33, convergence: float=1e-9, visc: float=None, Nm: float=6.5, l: float=0.05, visc_override: bool=False):
    """Function for single fixed point iterations for finding the steady state concentrations of ions in a diaphragm-separated two compartment flow cell
    
    Original code by Ben Charnay with help from Gage Wright and Rishi Agarwal
    This code by Gage Wright

    Args:
        ra (float): Anolyte flow rate in mL/min cm2
        rc (float): Catholyte flow rate in mL/min cm2
        electrolyte (str): Predefined electrolyte type, mixed/chloride/sulfate
        j (float): Current density, mA/cm2
        previous_condition (tuple, optional): (Anolyte dict, Catholyte dict) from a previous condition run or guess. Defaults to ().
        w (float, optional): Learning rate. Defaults to 0.33.
        convergence (float, optional): Maximum discrepancy between consecutive solutions to consider it solved. Defaults to 1e-9.
        visc (float, optional): Relative viscosity override. Defaults to None.
        Nm (float, optional): MacMullin Number. Defaults to 6.5.
        l (float, optional): Length of cell in cm. Defaults to 0.05.
        visc_override (bool, optional): Whether or not to override the predefined viscosity. Defaults to False.

    Returns:
        tuple (pd.DataFrame, dict, dict): Tracking DataFrame with all relevant simulation data and cA, cC dictionaries of concentrations from the final iteration
    """    

    F = 96485 # Faraday's constant
    
    # Convert flow rates to mL/second
    ra = ra/60
    rc = rc/60    
    
    ### Define values for the electrolyte
    electrolytes = {
        '3NaCl-pt75Na2SO4': [{'Na+':4.5,             # species and c_init / M in electrolyte
                              'Cl-':3,
                              'SO4--':0.75,
                              'NaSO4-':0,
                              'HSO4-':0},
                               ['NaSO4-', 'HSO4-'],    # adducts
                               1.74 * 1.162844799],    # relative dynamic viscosity
        '3NaCl': [{'Na+':3,
                   'Cl-':3},
                    [],
                    1.286/.9492], # JGW3-18
        'pt75Na2SO4': [{'Na+':1.5,
                        'SO4--':0.75,
                        'NaSO4-':0,
                        'HSO4-':0},
                         ['NaSO4-', 'HSO4-'],
                         1.53 * 1.09139248919907],
        }
    
    try:
        cA_init, adducts, predetermined_visc = electrolytes[electrolyte]
        cC_init = cA_init.copy()
        
    # Use the predefined viscosity even if visc is not None, except when visc_override is True
        if predetermined_visc is None:
            if visc is None:
                raise ValueError(f'Viscosity has not been predetermined for {electrolyte}')
            else: 
                pass
        elif not visc_override:
            visc = predetermined_visc
    except KeyError:    
        raise ValueError('Unrecognized Electrolyte')
    
    # Generate acid and base for the initial conditions
    cC_init['OH-'] = j/(F*rc) 
    cA_init['H+'] = j/(F*ra)
    
    # List ions
    species = list(set([ion for ion in list(cC_init.keys()) + list(cA_init.keys())]))
    cations = [spec for spec in species if '+' in spec]
    anions = [spec for spec in species if '-' in spec]
    ions = cations + anions
    charged_adducts = [spec for spec in adducts if '+' in spec or '-' in spec]
    
    # Diffusion Constants
    # Another empirically based approximation - not to correct OH or H diffusion constants for viscosity 
    D = {'Na+': 1.334e-5/visc,
         'Cl-': 2.032e-5/visc,
         'SO4--': 1.065e-5/visc,
         'H+': 9.311e-5,
         'OH-': 5.273e-5, 
         'HSO4-': 1.385e-5/visc,
        }
    # Conductivities, 10-4 m2 S mol-1, # Petr Vanysek
    lambda0 = {'Na+': 50.08,
               'Cl-': 76.31,
               'SO4--': 160,
               'H+': 349.65,
               'OH-': 189, 
               'HSO4-': 52,
               'NaSO4-': 30,
            }
    
    ### Array setup
    if previous_condition == None:
        cC = cC_init.copy()
        cA = cA_init.copy()
    else:
        # unpack everything from the previous run / guess
        cA, cC = previous_condition
    
    # initialize tracking dictionaries and add initial conditions
    trackingA, trackingC = {}, {}
    for spec in list(cC_init.keys()):
        trackingC[spec] = [cC_init[spec]]
    for spec in list(cA_init.keys()):
        trackingA[spec] = [cA_init[spec]]
        
    tracking = {'Anolyte Production': [np.nan], # Anolyte production in M
                'Catholyte Production': [np.nan], # Catholyte production in M
                'CE': [np.nan], # Current Efficiency
                'E0': [np.nan],
                'Catholyte pH': [np.nan],
                'Anolyte pH': [np.nan], 
                'Delta pH': [np.nan],  
                'Unbalanced Charge': [np.nan], # Unbalanced charge after diffusion computed from anolyte
                'kappa': [np.nan], # Effective conductivity
                }
    for ion in ions:
        tracking[f"t_{ion}"] = [np.nan]
        
    delta = 1 # Dummy value
    ### Solve condition
    while delta > convergence: # abs(max(Δ values)) > convergence:

        # Calculate charge imbalance
        UnbalA, UnbalC = 0, 0
        for ion in ions:
            if ion in list(cA.keys()):
                UnbalA += (cA[ion]*ion.count('+') - cA[ion]*ion.count('-'))
            if ion in list(cC.keys()):
                UnbalC += (cC[ion]*ion.count('+') - cC[ion]*ion.count('-'))
        
        # Add extra, ideal ions to balance charge in pyequion
        cA_extra, cC_extra = cA.copy(), cC.copy()
        if 'Cl-' in anions:
            cA_extra['Cl-'] += UnbalA
        else:
            cA_extra['Cl-'] = UnbalA
        if 'Na' not in electrolyte:
            if 'K+' in cations:
                cC_extra['K+'] -= UnbalC
        else:
            if 'Na+' in cations:
                cC_extra['Na+'] -= UnbalC
            else:
                cC_extra['Na+'] = -UnbalC
                
        # Solve catholyte and anolyte 
        # This function takes inputs in mM
        catholyte = pyequion.solve_solution({k: v * 1000 for k, v in cC_extra.items()}, activity_model_type='pitzer')
        anolyte = pyequion.solve_solution({k: v * 1000 for k, v in cA_extra.items()}, activity_model_type='pitzer')
        # catholyte.extend_result()
        # anolyte.extend_result()
        
        """
        Correct conductivities for activity. Only compute for ions that can electromigrate across the separator.
        The physical origin of this equation is for vehicular diffusion, so it approximately does not apply to protons.
        Empirically, this correction should be applied for hydroxide anyway, even though can can also participate in structural diffusion.
        """
        lambda_corr = {'H+': lambda0['H+']}
        for ion in ions:
            if ion in cations and 'H+' not in ion:
                lambda_corr[ion] = lambda0[ion] / visc
            elif ion in anions and 'OH-' not in ion:
                lambda_corr[ion] = lambda0[ion] / visc
            elif ion == 'OH-':
                lambda_corr[ion] = lambda0[ion]
        
        # Total conductivity
        kappa = 0
        for ion in ions:
            if ion in cations:
                kappa += anolyte.concentrations[ion] * lambda_corr[ion]
            elif ion in anions:
                kappa += catholyte.concentrations[ion] * lambda_corr[ion]
        
        ## Compute transference numbers, which are independent of diffusion
        transference = {}
        for ion in ions:
            if ion in cations:
                transference[ion] = anolyte.concentrations[ion] * lambda_corr[ion] / kappa
            elif ion in anions:
                transference[ion] = catholyte.concentrations[ion] * lambda_corr[ion] / kappa
        
        
        # Compute diffusion flux in mmol/sec
        diffusion_conc = {}
        for spec in species:
            if spec in list(D.keys()): # If no diffusion constant, ignore it
                diffusion_conc[spec] = D[spec] * (catholyte.concentrations[spec] - anolyte.concentrations[spec]) / (l * Nm)
            
        # Starting from initial concentrations, implement diffusion and manually approximate acid and base neutralization
        for spec in species:
            # Handle H+ and OH-, which neutralize each other
            if 'H+' in spec:
                cA[spec] = cA_init[spec] - ((-diffusion_conc[spec] + diffusion_conc['OH-']) / ra)
            elif 'OH-' in spec:
                cC[spec] = cC_init[spec] - ((diffusion_conc[spec] - diffusion_conc['H+']) / rc)
                
            # Handle Adducts
            elif 'HSO4-' in spec:
                cA['SO4--'] += diffusion_conc['HSO4-'] / ra
                cC['SO4--'] -= diffusion_conc['HSO4-'] / rc
                cA['H+'] += diffusion_conc['HSO4-'] / ra
                cC['OH-'] += diffusion_conc['HSO4-'] / rc
            elif 'NaSO4-' in spec:
                pass # No NaSO4- diffusion constant
                
            # Handle species that simply diffuse
            elif spec not in adducts:
                cA[spec] = cA_init[spec] + (diffusion_conc[spec] / ra)
                cC[spec] = cC_init[spec] - (diffusion_conc[spec] / rc)
                
                
        # Calculate charge imbalance
        UnbalA, UnbalC = 0, 0
        for ion in ions:
            if ion in list(cA.keys()):
                UnbalA += (cA[ion]*ion.count('+') - cA[ion]*ion.count('-')) * ra
            # if ion in list(cC.keys()):
            #     UnbalC += (cC[ion]*ion.count('+') - cC[ion]*ion.count('-')) * rc
        # UnbalC does not need computed since it is equal to UnbalA
        
        # Do transference
        i_t = UnbalA * F
        BalA = {}
        BalC = {}
        for ion in ions:
            if 'H+' in ion:
                BalA[ion] = cA[ion] - (((transference[ion] + transference['OH-']) * i_t) / (ra * F))
            elif 'OH-' in ion:
                BalC[ion] = cC[ion] - (((transference[ion] + transference['H+']) * i_t) / (rc * F))
            elif ion not in adducts:
                if ion in anions:
                    BalA[ion] = cA[ion] + (transference[ion] * i_t / (ra * F * ion.count('-')))
                    BalC[ion] = cC[ion] - (transference[ion] * i_t / (rc * F * ion.count('-')))
                elif ion in cations:
                    BalA[ion] = cA[ion] - (transference[ion] * i_t / (ra * F * ion.count('+')))
                    BalC[ion] = cC[ion] + (transference[ion] * i_t / (rc * F * ion.count('+')))
                    
        # Migrate charged adducts and sort into component species
        # Note that HSO4- cannot exist in the catholyte, so it cannot contribute to migration across the diaphragm
        for ion in charged_adducts:
            if 'NaSO4-' in ion:
                BalA['SO4--'] += (transference['NaSO4-'] * i_t) / (ra * F)
                BalC['SO4--'] -= (transference['NaSO4-'] * i_t) / (rc * F)
                BalA['Na+'] -= (transference['NaSO4-'] * i_t) / (ra * F)
                BalC['Na+'] -= (transference['NaSO4-'] * i_t) / (rc * F)
                    
            
        ## Learn from previous iteration 
        # Initialize new tracking dictionaries   
        cA = {}
        cC = {}
        for adduct in adducts:
            cA[adduct] = 0
            cC[adduct] = 0
            
        for spec in species:
            if spec in list(BalA.keys()):
                cA[spec] = round((w * BalA[spec]) + ((1-w) * trackingA[spec][-1]), 10)
            if spec in list(BalC.keys()):
                cC[spec] = round((w * BalC[spec]) + ((1-w) * trackingC[spec][-1]), 10)

        # Update tracking
        for spec in (cA.keys()):
            trackingA[spec].append(cA[spec])
        for spec in (cC.keys()):
            trackingC[spec].append(cC[spec])

        try:
            assert math.isclose(trackingC['OH-'][-1] / cC_init['OH-'], trackingA['H+'][-1] / cA_init['H+'], abs_tol=1e-2) # CEs must be identical
        except AssertionError:
            log.warning(f"Computed CEs do not match within 1%\nBase CE: {trackingC['OH-'][-1] / cC_init['OH-']}, Acid CE: {trackingA['H+'][-1] / cA_init['H+']}")
        
        tracking['Anolyte Production'].append(cA_init['H+'])
        tracking['Catholyte Production'].append(cC_init['OH-'])
        tracking['CE'].append(trackingC['OH-'][-1] / cC_init['OH-'])
        tracking['E0'].append((catholyte.pH - anolyte.pH)*0.0592)
        tracking['Catholyte pH'].append(catholyte.pH)
        tracking['Anolyte pH'].append(anolyte.pH)
        tracking['Delta pH'].append(catholyte.pH - anolyte.pH)
        tracking['Unbalanced Charge'].append(UnbalA)
        tracking['kappa'].append(kappa) # Effective conductivity
        for ion in ions:
            tracking['t_'+ion].append(transference[ion])
        
        # compute difference between this and previous iteration
        deltas = []
        for spec in (cA.keys()):
            deltas.append(trackingA[spec][-1] - trackingA[spec][-2])
        for spec in (cC.keys()):
            deltas.append(trackingC[spec][-1] - trackingC[spec][-2])
        delta = abs(max(deltas))
        
    
    ## label ions and return tracking data
    # relabel dictionary keys by compartment
    trackingA_labeled = {key+'A': trackingA[key] for key in trackingA.keys()}
    trackingC_labeled = {key+'C': trackingC[key] for key in trackingC.keys()}
    tracking = pd.concat([pd.DataFrame(tracking), pd.DataFrame(trackingA_labeled), pd.DataFrame(trackingC_labeled)], axis=1)
    
    log.debug(f"CE: {100*round(trackingC['OH-'][-1] / cC_init['OH-'],5)}")
    # log.debug(f"Unbalanced Charge: {UnbalA}")
    # conductances = {key: value * lambda_corr[key] for key, value in transference.items()}
    # log.debug(f"Conductance: {conductances}")
    # log.debug(f"Diffusion: {diffusion_conc} \n")
    
    return tracking, cA, cC # df, dict, dict
    
    
if __name__ == "__main__":
    previous = None
    for x in range(1,11):
        if x > 1:
            previous = (A, C)
        tracking, A, C = singleCondition(electrolyte='3NaCl-pt75Na2SO4', 
                                         ra=.1, rc=.1, 
                                         j=x*50, 
                                         previous_condition=previous)
        log.info(f"j: {x*50}, CE: {tracking.iloc[-1, tracking.columns.get_loc('CE')]}")

        
    tracking.to_csv('result.csv')
