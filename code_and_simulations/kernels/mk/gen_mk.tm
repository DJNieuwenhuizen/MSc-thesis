KPL/MK

gen_mk.tm
---------------------------------------------------------------------------------------------------

    This is the meta-kernel used in 'main_MEX-TGO_measurement_planning.py', calling general Solar System / Time / Mars related kernels.
    It was created on March 12, 2025 by Dominique Julianne Nieuwenhuizen (TU Delft)

---------------------------------------------------------------------------------------------------

    The names and contents of the kernels referenced by this meta-kernel are as follows:
    1. Generic Leapseconds Kernel (LSK)
        naif0012.tls.pc
    2. GM (gravitational constant times mass) values for the Sun, planets and planetary system barycenters (PCK):
        de-403-masses.tpc
    3. Generic Planetary Constants Kernel (PCK)
        pck00011.tpc
    4. Mars Satellite Ephemeris SPK, required for location of Mars CoM 499 (diff of ~20cm to Marc BC 4; covers 1900 Jan 04 to 2100 Jan 03)
        mar097.bsp
    5. Solar System Ephemeris SPK, current official planetary ephemeris (covers 1550 Jan 01 to 2650 Jan 22)
        de430.bsp 

---------------------------------------------------------------------------------------------------

\begindata

    PATH_VALUES       = ( 'C:\Users\donie\OneDrive - Delft University of Technology\Documenten+'
                           '\GitHub\thesis\code_and_simulations\kernels+' )

    PATH_SYMBOLS      = ( 'KERNELS' )

    KERNELS_TO_LOAD = (
                       '$KERNELS\lsk\naif0012.tls.pc'

                       '$KERNELS\pck\de-403-masses.tpc'
                       '$KERNELS\pck\pck00011.tpc'

                       '$KERNELS\spk\mar097.bsp'
                       '$KERNELS\spk\de430.bsp'
    )

\begintext

---------------------------------------------------------------------------------------------------

End of MK file.