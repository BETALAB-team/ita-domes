"""Run in an IDE to reproduce the 2021 electricity diminishing-returns analysis.

Uses the identical weighted cell-mean WCSS analysis as run_heating_elbow.py.
Saved appliance kWh estimates, archetypes and data splits remain unchanged.
Insulation is excluded. Output: revised_paper_outputs/electricity_elbow/2021.
The original Tukey comparison uses location x dwelling type, as in the original
electricity workflow; the new elbow retains all three current grouping factors.
"""
from run_heating_elbow import main


if __name__ == '__main__':
    main('electricity')
