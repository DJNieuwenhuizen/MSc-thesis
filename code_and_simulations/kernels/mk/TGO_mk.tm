KPL/MK

TGO_mk.tm
---------------------------------------------------------------------------------------------------

    This is the meta-kernel used in 'main_MEX-TGO_measurement_planning.py', calling TGO-specific kernels
    It was created on March 12, 2025 by Dominique Julianne Nieuwenhuizen (TU Delft)

---------------------------------------------------------------------------------------------------

    The names and contents of the kernels referenced by this meta-kernel are as follows:
    1. TGO S/C Trajectory SPK, used for long term planning operational Mars centric ephemeris (em16_tgo_flp_NNN_NN_YYYYMMDD_YYYYMMDD_vVV.bsp; only predicted data for the science phase):
        em16_tgo_flp_010_01_20220804_20310104_v01.bsp
    2. TGO S/C Trajectory SPKs, final files used for data analysis (em16_tgo_fsp_NNN_NN_YYYYMMDD_YYYYMMDD_vVV.bsp; contains predicted and reconstructed ephemeris for the science phase)
        em16_tgo_fsp_332_01_20240101_20240615_v01.bsp
        em16_tgo_fsp_333_01_20240109_20240622_v01.bsp
        em16_tgo_fsp_334_01_20240113_20240706_v01.bsp
        em16_tgo_fsp_335_01_20240113_20240706_v01.bsp
        em16_tgo_fsp_336_01_20240129_20240713_v01.bsp
        em16_tgo_fsp_337_01_20240205_20240720_v01.bsp
        em16_tgo_fsp_338_01_20240212_20240727_v01.bsp
        em16_tgo_fsp_339_01_20240220_20240803_v01.bsp
        em16_tgo_fsp_340_01_20240226_20240810_v01.bsp
        em16_tgo_fsp_341_01_20240304_20240817_v01.bsp
        em16_tgo_fsp_342_01_20240312_20240824_v01.bsp
        em16_tgo_fsp_343_01_20240319_20240831_v01.bsp
        em16_tgo_fsp_344_01_20240325_20240907_v01.bsp
        em16_tgo_fsp_345_01_20240402_20240914_v01.bsp
        em16_tgo_fsp_346_01_20240408_20240921_v01.bsp
        em16_tgo_fsp_347_01_20240416_20240928_v01.bsp
        em16_tgo_fsp_348_01_20240422_20241005_v01.bsp
        em16_tgo_fsp_349_01_20240429_20241012_v01.bsp
        em16_tgo_fsp_350_01_20240506_20241019_v01.bsp
        em16_tgo_fsp_351_01_20240515_20241026_v01.bsp
        em16_tgo_fsp_352_01_20240520_20241102_v01.bsp
        em16_tgo_fsp_353_01_20240528_20241109_v01.bsp
        em16_tgo_fsp_354_01_20240603_20241116_v01.bsp
        em16_tgo_fsp_355_01_20240610_20241123_v01.bsp
        em16_tgo_fsp_356_01_20240617_20241130_v01.bsp
        em16_tgo_fsp_357_01_20240624_20241207_v01.bsp
        em16_tgo_fsp_358_01_20240703_20241214_v01.bsp
        em16_tgo_fsp_359_01_20240708_20241221_v01.bsp
        em16_tgo_fsp_360_01_20240715_20241228_v01.bsp
        em16_tgo_fsp_361_01_20240722_20250104_v01.bsp
        em16_tgo_fsp_362_01_20240729_20250111_v01.bsp
        em16_tgo_fsp_363_01_20240806_20250201_v01.bsp
        em16_tgo_fsp_364_01_20240806_20250201_v01.bsp
        em16_tgo_fsp_365_01_20240813_20250201_v01.bsp
        em16_tgo_fsp_366_01_20240813_20250201_v01.bsp
        em16_tgo_fsp_367_01_20240903_20250215_v01.bsp
        em16_tgo_fsp_368_01_20240909_20250222_v01.bsp
        em16_tgo_fsp_369_01_20240916_20250301_v01.bsp
        em16_tgo_fsp_370_01_20240923_20250308_v01.bsp
        em16_tgo_fsp_371_01_20240930_20250315_v01.bsp
        em16_tgo_fsp_372_01_20241008_20250322_v01.bsp
        em16_tgo_fsp_373_01_20241014_20250329_v01.bsp
        em16_tgo_fsp_374_01_20241021_20250405_v01.bsp
        em16_tgo_fsp_375_01_20241028_20250412_v01.bsp
        em16_tgo_fsp_376_01_20241104_20250419_v01.bsp
        em16_tgo_fsp_377_01_20241111_20250426_v01.bsp
        em16_tgo_fsp_378_01_20241118_20250503_v01.bsp
        em16_tgo_fsp_379_01_20241125_20250510_v01.bsp
        em16_tgo_fsp_380_01_20241202_20250517_v01.bsp
        em16_tgo_fsp_381_01_20241209_20250524_v01.bsp
        em16_tgo_fsp_382_01_20241209_20250524_v01.bsp
        em16_tgo_fsp_383_01_20241217_20250621_v01.bsp
        em16_tgo_fsp_384_01_20241230_20250621_v01.bsp
        em16_tgo_fsp_385_01_20250106_20250621_v01.bsp
        em16_tgo_fsp_386_01_20250113_20250628_v01.bsp
        em16_tgo_fsp_387_01_20250120_20250705_v01.bsp
        em16_tgo_fsp_388_01_20250127_20250712_v01.bsp
        em16_tgo_fsp_389_01_20250130_20250802_v01.bsp
        em16_tgo_fsp_390_01_20250130_20250802_v01.bsp
        em16_tgo_fsp_391_01_20250218_20250802_v01.bsp
        em16_tgo_fsp_392_01_20250224_20250809_v01.bsp
        em16_tgo_fsp_393_01_20250303_20250816_v01.bsp
        em16_tgo_fsp_394_01_20250310_20250823_v01.bsp
        em16_tgo_fsp_395_01_20250317_20250830_v01.bsp
        ... Should be kept updated as new versions are released on a frequent basis
    3. Trace Gas Orbiter (TGO) Spacecraft Frames Kernel, version 2.7
        em16_tgo_v27.tf

---------------------------------------------------------------------------------------------------

\begindata

    PATH_VALUES       = ( 'C:\Users\donie\OneDrive - Delft University of Technology\Documenten+'
                           '\GitHub\thesis\code_and_simulations\kernels+' )

    PATH_SYMBOLS      = ( 'KERNELS' )

    KERNELS_TO_LOAD = (
                       '$KERNELS\spk\em16_tgo_flp_010_01_20220804_20310104_v01.bsp'

                       '$KERNELS\spk\em16_tgo_fsp_332_01_20240101_20240615_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_333_01_20240109_20240622_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_334_01_20240113_20240706_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_335_01_20240113_20240706_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_336_01_20240129_20240713_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_337_01_20240205_20240720_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_338_01_20240212_20240727_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_339_01_20240220_20240803_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_340_01_20240226_20240810_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_341_01_20240304_20240817_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_342_01_20240312_20240824_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_343_01_20240319_20240831_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_344_01_20240325_20240907_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_345_01_20240402_20240914_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_346_01_20240408_20240921_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_347_01_20240416_20240928_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_348_01_20240422_20241005_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_349_01_20240429_20241012_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_350_01_20240506_20241019_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_351_01_20240515_20241026_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_352_01_20240520_20241102_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_353_01_20240528_20241109_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_354_01_20240603_20241116_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_355_01_20240610_20241123_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_356_01_20240617_20241130_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_357_01_20240624_20241207_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_358_01_20240703_20241214_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_359_01_20240708_20241221_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_360_01_20240715_20241228_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_361_01_20240722_20250104_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_362_01_20240729_20250111_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_363_01_20240806_20250201_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_364_01_20240806_20250201_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_365_01_20240813_20250201_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_366_01_20240813_20250201_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_367_01_20240903_20250215_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_368_01_20240909_20250222_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_369_01_20240916_20250301_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_370_01_20240923_20250308_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_371_01_20240930_20250315_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_372_01_20241008_20250322_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_373_01_20241014_20250329_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_374_01_20241021_20250405_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_375_01_20241028_20250412_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_376_01_20241104_20250419_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_377_01_20241111_20250426_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_378_01_20241118_20250503_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_379_01_20241125_20250510_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_380_01_20241202_20250517_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_381_01_20241209_20250524_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_382_01_20241209_20250524_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_383_01_20241217_20250621_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_384_01_20241230_20250621_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_385_01_20250106_20250621_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_386_01_20250113_20250628_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_387_01_20250120_20250705_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_388_01_20250127_20250712_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_389_01_20250130_20250802_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_390_01_20250130_20250802_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_391_01_20250218_20250802_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_392_01_20250224_20250809_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_393_01_20250303_20250816_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_394_01_20250310_20250823_v01.bsp'
                       '$KERNELS\spk\em16_tgo_fsp_395_01_20250317_20250830_v01.bsp'

                       '$KERNELS\fk\em16_tgo_v27.tf'

    )

\begintext

---------------------------------------------------------------------------------------------------

End of MK file.