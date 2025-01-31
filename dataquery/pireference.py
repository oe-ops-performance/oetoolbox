import itertools
import os
import pandas as pd

from oetoolbox.utils import oemeta

af_dict = oemeta.data
allsolarsites = list(af_dict["AF_Solar_V3"])
solarsitesinpi = [site for site in allsolarsites if af_dict["AF_Solar_V3"][site] != {}]

allwindsites = list(af_dict["AF_Wind_V3"])
allgassites = list(af_dict["AF_Gas_V3"])
all_pi_sites = allsolarsites + allwindsites + allgassites

atlas_sitenames = [
    "AZ1",
    "FL1",
    "FL4",
    "GA3",
    "GA4",
    "Grand View East",
    "Grand View West",
    "Maplewood 1",
    "Maplewood 2",
    "MS3",
    "Sweetwater",
    "Three Peaks",
]


"""TEMP"""
sweetwater_trk_ref = {
    "B1.01": {
        "NCU1": [*range(106, 117), *range(201, 229), *range(301, 329)],
        "NCU2": [*range(401, 429), *range(501, 529)],
    },
    "B1.02": {
        "NCU1": [*range(101, 116), *range(201, 216), *range(301, 333), *range(807, 808)],
        "NCU2": [*range(401, 433), *range(501, 514), *range(601, 614)],
    },
    "B1.03": {
        "NCU1": [*range(110, 113), *range(210, 217), *range(301, 333), *range(401, 429)],
        "NCU2": [*range(501, 529), *range(601, 629)],
    },
    "B1.04": {
        "NCU1": [*range(107, 134), *range(205, 234)],
        "NCU2": [*range(303, 334), *range(401, 434)],
    },
    "B1.05": {
        "NCU1": [*range(101, 150), *range(201, 221)],
        "NCU2": [*range(221, 250), *range(301, 314), *range(401, 412)],
    },
    "B1.06": {
        "NCU1": [*range(101, 125), *range(201, 225), *range(302, 314)],
        "NCU2": [*range(314, 338), *range(402, 438)],
    },
    "B1.07": {
        "NCU1": [*range(101, 131), *range(201, 231)],
        "NCU2": [*range(302, 332), *range(402, 432)],
    },
    "B1.08": {
        "NCU1": [*range(107, 134), *range(205, 234), *range(913, 914)],
        "NCU2": [*range(303, 335), *range(401, 435)],
    },
    "B1.09": {
        "NCU1": [*range(101, 119), *range(201, 219), *range(307, 320)],
        "NCU2": [*range(320, 348), *range(407, 447)],
    },
    "B1.10": {
        "NCU1": [*range(101, 129), *range(201, 229)],
        "NCU2": [*range(229, 235), *range(313, 335), *range(413, 429)],
    },
    "B1.11": {
        "NCU1": [*range(117, 132), *range(201, 232), *range(301, 317)],
        "NCU2": [*range(317, 323), *range(403, 423), *range(503, 517)],
    },
    "B1.12": {
        "NCU1": [*range(101, 158), *range(225, 243), *range(945, 946)],
        "NCU2": [*range(244, 258), *range(351, 372)],
    },
    "B1.13": {
        "NCU1": [*range(101, 126), *range(201, 226)],
        "NCU2": [*range(301, 326), *range(401, 426)],
    },
    "B1.14": {
        "NCU1": [*range(101, 126), *range(201, 226)],
        "NCU2": [*range(301, 326), *range(401, 426)],
    },
    "B1.15": {
        "NCU1": [*range(101, 127), *range(201, 227)],
        "NCU2": [*range(303, 327), *range(403, 427)],
    },
    "B1.16": {
        "NCU1": [*range(101, 137), *range(203, 230)],
        "NCU2": [*range(230, 237), *range(313, 339), *range(430, 439)],
    },
    "B2.01": {
        "NCU1": [*range(101, 124), *range(201, 228)],
        "NCU2": [*range(128, 148), *range(231, 259)],
    },
    "B2.02": {
        "NCU1": [*range(101, 125), *range(201, 231), *range(911, 912)],
        "NCU2": [*range(301, 332), *range(414, 417), *range(501, 517)],
    },
    "B2.03": {"NCU1": [*range(101, 149)], "NCU2": [*range(201, 253)]},
    "B2.04": {
        "NCU1": [*range(101, 131), *range(201, 222)],
        "NCU2": [*range(222, 231), *range(310, 331), *range(410, 431)],
    },
    "B2.05": {
        "NCU1": [*range(101, 120), *range(201, 224), *range(301, 316)],
        "NCU2": [*range(316, 330), *range(401, 433)],
    },
    "B2.06": {
        "NCU1": [*range(107, 130), *range(207, 230)],
        "NCU2": [*range(301, 329), *range(401, 429)],
    },
    "B2.07": {
        "NCU1": [*range(102, 127), *range(202, 227)],
        "NCU2": [*range(301, 327), *range(401, 427), *range(950, 951)],
    },
    "B2.08": {
        "NCU1": [*range(101, 121), *range(201, 225)],
        "NCU2": [*range(301, 329), *range(401, 433)],
    },
    "B2.09": {
        "NCU1": [*range(101, 131), *range(201, 223)],
        "NCU2": [*range(223, 239), *range(305, 325), *range(409, 425)],
    },
    "B2.10": {
        "NCU1": [*range(112, 136), *range(212, 244)],
        "NCU2": [*range(301, 327), *range(405, 427)],
    },
    "B2.11": {
        "NCU1": [*range(119, 130), *range(219, 230), *range(301, 328), *range(405, 419)],
        "NCU2": [*range(510, 528), *range(611, 628), *range(714, 728)],
    },
    "B2.12": {
        "NCU1": [*range(103, 119), *range(203, 227), *range(301, 317)],
        "NCU2": [*range(401, 417), *range(501, 517), *range(601, 617)],
    },
    "B2.13": {
        "NCU1": [*range(101, 121), *range(201, 229)],
        "NCU2": [*range(301, 332), *range(401, 432), *range(858, 859)],
    },
    "B2.14": {
        "NCU1": [*range(101, 131), *range(201, 231)],
        "NCU2": [*range(301, 328), *range(409, 429)],
    },
    "B2.15": {
        "NCU1": [*range(101, 128), *range(201, 228)],
        "NCU2": [*range(301, 328), *range(401, 428), *range(962, 963)],
    },
    "B2.16": {
        "NCU1": [*range(112, 114), *range(212, 214), *range(301, 341), *range(410, 426)],
        "NCU2": [*range(426, 441), *range(519, 540), *range(628, 641)],
    },
}

sweetwater_trk_att_dict = {
    block_: (
        list(
            itertools.chain.from_iterable(
                [[f"NCU1.m{n}_pos", f"NCU1.m{n}_sp"] for n in block_dict["NCU1"]]
            )
        )
        + list(
            itertools.chain.from_iterable(
                [[f"NCU2.m{n}_pos", f"NCU2.m{n}_sp"] for n in block_dict["NCU2"]]
            )
        )
    )
    for block_, block_dict in sweetwater_trk_ref.items()
}

threepeaks_trk_ref = {
    "B101.NCU1": [*range(101, 136), *range(201, 219)],
    "B101.NCU2": [*range(136, 164), *range(219, 247)],
    "B102.NCU1": [*range(101, 127), *range(201, 226)],
    "B102.NCU2": [*range(128, 155), *range(227, 254)],
    "B103.NCU1": [*range(101, 127), *range(201, 226)],
    "B103.NCU2": [*range(128, 155), *range(227, 254)],
    "B104.NCU1": [*range(101, 127), *range(201, 226)],
    "B104.NCU2": [*range(128, 155), *range(227, 254)],
    "B105.NCU1": [*range(101, 127), *range(201, 226)],
    "B105.NCU2": [*range(128, 155), *range(227, 254)],
    "B106.NCU1": [*range(101, 127), *range(201, 227)],
    "B106.NCU2": [*range(301, 328), *range(401, 427)],
    "B107.NCU1": [*range(101, 127), *range(201, 226)],
    "B107.NCU2": [*range(128, 155), *range(227, 254)],
    "B108.NCU1": [*range(101, 127), *range(201, 226)],
    "B108.NCU2": [*range(128, 155), *range(227, 254)],
    "B109.NCU1": [*range(101, 127), *range(201, 226)],
    "B109.NCU2": [*range(128, 155), *range(227, 254)],
    "B201.NCU1": [*range(101, 127), *range(201, 227)],
    "B201.NCU2": [*range(128, 154), *range(228, 255)],
    "B202.NCU1": [*range(101, 127), *range(201, 227)],
    "B202.NCU2": [*range(128, 154), *range(228, 255)],
    "B203.NCU1": [*range(101, 127), *range(201, 227)],
    "B203.NCU2": [*range(128, 154), *range(228, 255)],
    "B204.NCU1": [*range(101, 127), *range(201, 227)],
    "B204.NCU2": [*range(128, 154), *range(228, 255)],
    "B205.NCU1": [*range(101, 127), *range(201, 227)],
    "B205.NCU2": [*range(128, 154), *range(228, 255)],
    "B206.NCU1": [*range(101, 127), *range(201, 227)],
    "B206.NCU2": [*range(128, 154), *range(228, 255)],
    "B207.NCU1": [*range(101, 127), *range(201, 227)],
    "B207.NCU2": [*range(128, 154), *range(228, 255)],
    "B208.NCU1": [*range(101, 127), *range(201, 227)],
    "B208.NCU2": [*range(128, 154), *range(228, 255)],
    "B209.NCU1": [*range(101, 127), *range(201, 227)],
    "B209.NCU2": [*range(128, 154), *range(228, 255)],
    "B301.NCU1": [*range(101, 127), *range(201, 226)],
    "B301.NCU2": [*range(128, 154), *range(227, 253)],
    "B302.NCU1": [*range(101, 127), *range(201, 226)],
    "B302.NCU2": [*range(128, 154), *range(227, 253)],
    "B303.NCU1": [*range(101, 127), *range(201, 226)],
    "B303.NCU2": [*range(128, 154), *range(227, 253)],
    "B304.NCU1": [*range(101, 127), *range(201, 226)],
    "B304.NCU2": [*range(128, 154), *range(227, 253)],
    "B305.NCU1": [*range(101, 127), *range(201, 226)],
    "B305.NCU2": [*range(128, 154), *range(227, 253)],
    "B306.NCU1": [*range(101, 127), *range(201, 226)],
    "B306.NCU2": [*range(128, 154), *range(227, 253)],
    "B307.NCU1": [*range(101, 127), *range(201, 226)],
    "B307.NCU2": [*range(128, 154), *range(227, 253)],
    "B308.NCU1": [*range(101, 127), *range(201, 226)],
    "B308.NCU2": [*range(128, 154), *range(227, 253)],
    "B309.NCU1": [*range(101, 127), *range(201, 226)],
    "B309.NCU2": [*range(128, 154), *range(227, 253)],
    "B401.NCU1": [*range(101, 127), *range(201, 227)],
    "B401.NCU2": [*range(128, 153), *range(228, 254)],
    "B402.NCU1": [*range(101, 127), *range(201, 227)],
    "B402.NCU2": [*range(128, 153), *range(228, 254)],
    "B403.NCU1": [*range(101, 127), *range(201, 227)],
    "B403.NCU2": [*range(128, 153), *range(228, 254)],
    "B404.NCU1": [*range(101, 127), *range(201, 227)],
    "B404.NCU2": [*range(128, 153), *range(228, 254)],
    "B405.NCU1": [*range(101, 127), *range(201, 227)],
    "B405.NCU2": [*range(128, 153), *range(228, 254)],
    "B406.NCU1": [*range(101, 127), *range(201, 227)],
    "B406.NCU2": [*range(128, 153), *range(228, 254)],
    "B407.NCU1": [*range(101, 127), *range(201, 227)],
    "B407.NCU2": [*range(128, 153), *range(228, 254)],
    "B408.NCU1": [*range(101, 127), *range(201, 227)],
    "B408.NCU2": [*range(128, 153), *range(228, 254)],
    "B409.NCU1": [*range(101, 127), *range(201, 227)],
    "B409.NCU2": [*range(128, 153), *range(228, 254)],
}
threepeaks_trk_att_dict = {
    trk_: list(itertools.chain.from_iterable([[f"m{n}_pos", f"m{n}_sp"] for n in rng_list]))
    for trk_, rng_list in threepeaks_trk_ref.items()
}


"""
REFERENCE DICTIONARY
>> pi query metadata containing assets/attributes for monthly query
structure = {
    'Proj Name': {
  (new) 'acmod atts'     : [list, ac module related attributes for inverter assets (no sub-assets)],
  (new) 'acmod sub atts' : [list, ac module related attributes for combiners (inverter sub-assets)],
        'inv atts'       : [list, cmb box atts when atts of inverter assets (no sub-assets)],
        'inv sub atts'   : [list, cmb box atts when combiners are sub-assets of inv assets],
        'met ids'        : [list, met station ids (if subset of available)],
        'metsta atts'    : [list, met station atts incl. POA, module temp, and wind speed],
        'ppc atts'       : [list, ppc curtailment atts incl. signal and setpoint],
        'trk sub atts'   : [list, tracker motor atts (trk motors are sub-assets of trk assets)],
        'meter pipoint'  : pipoint for meter power data (dtype: str),
        'MW adjust'      : scaling factor to get meter data in MWh (dtype: float),
    }
}
"""
pqmeta = {
    "Adams East": {
        "acmod atts": [f"M{n}.Sts.Running" for n in range(1, 5)],
        "inv atts": [f"DCCT0{n:02d}.Sts.I_A" for n in range(1, 20)],
        "metsta atts": {
            "P003": [f"MST00{n}.Sts.Temp_C" for n in range(1, 5)],
            "P006": ["OE.POA", "WS001.Sts.WindSpdAvg_mps", "WS001.Sts.GHIrr_Wm2"],
            "P013": (
                ["OE.POA", "WS001.Sts.WindSpdAvg_mps", "WS001.Sts.GHIrr_Wm2"]
                + [f"MST00{n}.Sts.Temp_C" for n in range(2, 5)]
            ),
        },
        "trk sub atts": (
            [f"TA{n:02d}MtrI_A" for n in range(1, 11)]
            + [f"TA{n:02d}TiltAng_Deg" for n in range(1, 11)]
        ),
        "meter pipoint": "ADMES.AES1_S001_SL.AES1_S001.PMCAISOPri.Sts.P_kW",
        "MW adjust": 1,  # -.001,
    },
    "Alamo": {
        # 'acmod atts': [],
        "inv atts": [f"DC_CURRENT_{n:02d}" for n in range(1, 13)],
        "metsta atts": [
            "OE.POA",
            "OE.Wind_Speed",
            "MODULETEMPERATURE1",
            "MODULETEMPERATURE2",
            "IRRADIANCE_GHI",
        ],
        "trk sub atts": ["OE.AvgPos_TM", "TRKR_SETPOINT"],
        "meter pipoint": "ALAMO.AL_RIG.01.ION8650_KW_TOT",
        "MW adjust": 0.001,
    },
    "AZ1": {
        # 'acmod atts': [],
        "metsta atts": ["OE.POA", "WS_1.BOM_1.Temp", "WS_1.BOM_2.Temp", "WS_1.WS_1.Wind_Speed"],
        "trk atts": (
            [f"Tracker_{n}.Position" for n in range(1, 5)]
            + [f"Tracker_{n}.Setpoint" for n in range(1, 5)]
        ),
        "meter pipoint": "ARZO1.Sub_1.Meter_1.Power",
        "MW adjust": 1,
    },
    "Azalea": {
        # 'acmod atts': [],
        "inv atts": [f"CMB{n}" for n in range(1, 12)],
        "metsta atts": {
            "Met1": ["OE.Wind_Speed", "GHI1_3S"],
            "Met2": ["OE.POA", "OE.Wind_Speed", "GHI1_3S"],
            "Z02": [f"PLC_MTMP{n}_3S" for n in range(1, 7)],
        },
        "meter pipoint": "AZLEA.ION01.AZL1_ION01_KW",
        "MW adjust": -0.001,
    },
    "Camelot": {
        # 'acmod atts': [],
        "inv atts": [f"CMB{n}_INDCCURR" for n in range(1, 17)],
        "met ids": ["WS02", "WS03"],
        "metsta atts": [
            "OE.POA",
            "OE.Wind_Speed",
            "MODBPTEMP1_C",
            "MODBPTEMP2_C",
            "IRRADGLOBALHOR",
        ],
        "trk sub atts": ["OE.AvgPos_TM", "SP"],
        "meter pipoint": "CMLOT.CAM_RIG_KW_TOT",
        "MW adjust": 1,
    },
    "Catalina II": {
        # 'acmod atts': [],
        "acmod atts": ["MODSTATE", "MODTEMP", "MODTRIP"],
        "metsta atts": ["OE.POA", "PNLTMP1", "PNLTMP2", "PYRIRR02_GHI", "PYRIRR03_GHI"],
        "trk atts": ["OE.AvgPos", "SETPOS"],
        "meter pipoint": "CATA2.CAT2_HMI.ION_8650_1_AI_015_kW_tot_CAISO",
        "MW adjust": -0.001,
    },
    "CID": {
        # 'acmod atts': [],
        "inv atts": [f"DCCT0{n:02d}.Sts.I_A" for n in range(1, 10)],
        "metsta atts": {
            "METB1P3": ["OE.POA", "PnlTemp_CC", "OE.Wind_Speed", "GHIrr_Wm2"],
            "METB1P7": ["PnlTemp_CC", "GHIrr_Wm2"],
        },
        "trk sub atts": (
            [f"TA{n:02d}MtrI_A" for n in range(1, 13)]
            + [f"TA{n:02d}TiltAng_Deg" for n in range(1, 13)]
        ),
        "meter pipoint": "Solar Assets.CID.Meter.OE.MeterMW",
        "MW adjust": 1,
    },
    "Columbia II": {
        # 'acmod atts': [],
        "inv atts": [f"CMB{n}_INDCCURR" for n in range(1, 17)],
        "metsta atts": {
            "WS02": ["OE.POA", "OE.Wind_Speed", "MODBPTEMP1_C", "IRRADGLOBALHOR"],
            "WS03": ["OE.POA", "MODBPTEMP1_C", "MODBPTEMP2_C", "IRRADGLOBALHOR"],
        },
        "trk sub atts": ["OE.AvgPos_TM", "SP"],
        "meter pipoint": "COLM2.C2_RIG_KW_TOT",
        "MW adjust": 1,
    },
    "Comanche": {
        # 'acmod atts': [],
        "inv atts": [f"CT_{n}" for n in range(1, 18)],
        "metsta atts": {
            "ENV06": ["OE.POA", "OE.Wind_Speed", "TMPR_MDL1", "TMPR_MDL2", "TMPR_MDL3", "GHI_1"],
            "ENV16": ["OE.POA", "TMPR_MDL1", "TMPR_MDL2", "TMPR_MDL3", "GHI_1"],
            "ENV27": ["OE.POA"],
            "ENV43": ["OE.POA", "TMPR_MDL1", "TMPR_MDL2", "TMPR_MDL3"],
            "ENV50": ["OE.POA", "TMPR_MDL1", "TMPR_MDL2", "TMPR_MDL3", "GHI_1"],
            "ENV69": ["ch_2_Irradiance"],
            "ENV72": ["OE.POA"],
        },
        "ppc atts": ["xcel_MW_cmd_request", "xcel_MW_setpoint"],
        "trk atts": ["OE.AvgPos", "Setpoint_Average", "stow"],
        "meter pipoint": "CMNCH:SSN01.MTR001.P_ac",
        "MW adjust": 0.001,
    },
    "CW-Corcoran": {
        # 'acmod atts': [],
        "inv atts": [f"DCCT0{n:02d}.Sts.I_A" for n in range(1, 11)],
        "metsta atts": {
            "P002": ["WS001.Sts.WindSpdAvg_mps"] + [f"MST00{n}.Sts.Temp_C" for n in range(1, 5)],
            "P006": [
                "OE.POA",
                "WS001.Sts.WindSpdAvg_mps",
                "WS001.Sts.GHIrr_Wm2",
                "MST003.Sts.Temp_C",
            ],
        },
        "trk sub atts": (
            [f"TA{n:02d}MtrI_A" for n in range(1, 13)]
            + [f"TA{n:02d}TiltAng_Deg" for n in range(1, 13)]
        ),
        "meter pipoint": "CCCRS.CCS1_S001.PMCAISOPRI.Sts.P_kW",
        "MW adjust": -0.001,
    },
    "CW-Goose Lake": {
        # 'acmod atts': [],
        "inv atts": [f"DCCT0{n:02d}.Sts.I_A" for n in range(1, 11)],
        "metsta atts": ["OE.POA", "PnlTemp_C", "WindSpdAvg_mps", "GHIrr_Wm2"],
        "trk sub atts": (
            [f"TA{n:02d}MtrI_A" for n in range(1, 13)]
            + [f"TA{n:02d}TiltAng_Deg" for n in range(1, 13)]
        ),
        "meter pipoint": "GOSLK.CGL1_S001.PMCAISOPRI.Sts.P_kW",
        "MW adjust": -0.001,
    },
    "CW-Marin": {
        "metsta atts": ["OE.POA", "PnlTemp2_C", "AirTemp_C", "GHIrr_Wm2", "WindSpdAvg_mps"],
        "meter pipoint": "MARIN.GEN_METER.Watts_3-Ph_Total",
        "MW adjust": -0.000001,
    },
    "FL1": {
        "metsta atts": ["OE.POA", "OE.ModuleTemp"],
        "meter pipoint": "FL1TY.PLC1.S_GRPTot_COUNT.PTOT",
        "MW adjust": 1,
    },
    "FL4": {
        "acmod atts": ["Mod_Num_Run", "Mods_Run"],
        "metsta atts": ["OE.POA", "PoA_2.Irr", "GHI_1.Irr", "WS_1.Wind_Speed"],
        "meter pipoint": "TALLA.Sub_1.Meter_1.Power",
        "MW adjust": 1,
    },
    "GA3": {
        "inv atts": [f"RCB_1_Current_{n}" for n in range(1, 17)],
        "metsta atts": {
            "WS1": ["BOM_2_Temp", "BOM_3_Temp"],
            "WS2": ["OE.POA", "GHI_1_Irr"],
            "WS3": ["OE.POA", "BOM_1_Temp", "BOM_2_Temp", "BOM_3_Temp", "GHI_1_Irr"],
            "WS4": ["OE.POA", "BOM_1_Temp", "BOM_3_Temp", "GHI_1_Irr"],
        },
        "meter pipoint": "GA3TW.Sub_1.Meter_1.Power",
        "MW adjust": 1,
    },
    "GA4": {
        # 'acmod atts': [],
        "acmod atts": ["Sts.RunningModules"] + [f"Module 00{n}.Sts.P_kW" for n in range(1, 7)],
        "inv atts": [f"Sts.DCCT.DCCT0{n:02d}_A" for n in range(1, 41)],
        "metsta atts": {
            "B1P03": ["OE.POA", "MSTemp001_C", "GHIrr_Wm2", "WindSpd_mps"],
            "B2P06": ["OE.POA", "MSTemp001_C", "GHIrr_Wm2", "WindSpd_mps"],
            "B3P03": ["OE.POA", "MSTemp001_C", "GHIrr_Wm2", "WindSpd_mps"],
            "B5P10": ["OE.POA", "GHIrr_Wm2", "WindSpd_mps"],
            "B6P02": ["OE.POA", "GHIrr_Wm2", "WindSpd_mps"],
        },
        "trk atts": ["Sts.CmdAng_Deg", "Sts.MeasAng_Deg"],
        "meter pipoint": "TWIGS.Substation.SEL3530.Meters.MTR_FSREV1.Sts.P_MW",
        "MW adjust": -0.000001,
    },
    "Grand View East": {
        # 'acmod atts': [],
        "inv atts": [f"DCamps_{n:02d}" for n in range(1, 25)],
        "metsta atts": ["OE.POA", "GHI_1", "Wind_Speed"] + [f"BOM_Temp_{n}" for n in range(1, 4)],
        "ppc atts": ["Gens_Faulted", "Gens_Gross_KW", "Gens_Offline", "Meter_KW", "Power_Limit_SP"],
        "meter pipoint": "GRDVE.meters.DNP.owner.KW",
        "MW adjust": 0.001,
    },
    "Grand View West": {
        # 'acmod atts': [],
        "inv atts": [f"DCamps_{n:02d}" for n in range(1, 25)],
        "metsta atts": {
            "B106": ["OE.POA", "GHI_1", "Wind_Speed"] + [f"BOM{n}" for n in range(1, 4)],
            "B204": ["OE.POA", "GHI_1", "Wind_Speed"] + [f"BOM{n}" for n in range(1, 4)],
            "B210": ["OE.POA"],
        },
        "ppc atts": ["Gens_Faulted", "Gens_Gross_KW", "Gens_Offline", "Meter_KW", "Power_Limit_SP"],
        "meter pipoint": "GRDVW.meters.DNP.owner.KW",
        "MW adjust": 0.001,
    },
    "Imperial Valley": {
        # 'acmod atts': [],
        "inv sub atts": [f"CURRENT_STRING_{n}" for n in range(1, 17)],
        "metsta atts": ["OE.Wind_Speed", "IRRADIANCE_1_POA", "TEMP_C"],
        "meter pipoint": "IMPVA.SS_PQ1_W3",
        "MW adjust": 1,
    },
    "Indy I": {
        # 'acmod atts': [],
        "inv atts": [f"Combiner_Current_{n}" for n in range(1, 12)],
        "metsta atts": {
            "Met01": [f"OE.Module_Temp{n}" for n in range(1, 6)]
            + ["POA_3_Sec_Sample_POA1", "GHI_3_Sec_Sample_GHI1"],
            "Met02": ["POA_3_Sec_Sample_POA1", "GHI_3_Sec_Sample_GHI1"],
        },
        "meter pipoint": "INDY1:ION01.ION01.IDY1_ION01_KW",
        "MW adjust": -0.001,
    },
    "Indy II": {
        # 'acmod atts': [],
        "inv atts": [f"Combiner_Current_{n}" for n in range(1, 12)],
        "metsta atts": {
            "MET01": ["POA_3_Sec_Sample_POA1", "GHI_3_Sec_Sample_GHI1"],
            "MET02": (
                [f"OE.Module_Temp{n}" for n in range(2, 5)]
                + ["POA_3_Sec_Sample_POA1", "Wind_Speed_3_Sec_Sample", "GHI_3_Sec_Sample_GHI1"]
            ),
        },
        "meter pipoint": "INDY2:ION01.ION01.IDY2_ION01_KW",
        "MW adjust": -0.001,
    },
    "Indy III": {
        # 'acmod atts': [],
        "inv atts": [f"Combiner_Current_{n}" for n in range(1, 12)],
        "metsta atts": {
            "MET01": [f"OE.Module_Temp{n}" for n in range(1, 5)] + ["POA_3_Sec_Sample_POA1"],
            "MET02": ["POA_3_Sec_Sample_POA1", "GHI_3_Sec_Sample_GHI1"],
        },
        "meter pipoint": "IDY3_ION01_KW",
        "MW adjust": -0.001,
    },
    "Kansas": {
        "acmod atts": [f"M{n}.Sts.Running" for n in range(1, 5)],
        "inv atts": [f"DCCT0{n:02d}.Sts.I_A" for n in range(1, 21)],
        "metsta atts": {
            "Met13": ["OE.POA", "PnlTemp_C", "WindSpdAvg_mps", "GHIrr_Wm2"],
            "Met4": ["PnlTemp_C", "WindSpdAvg_mps", "GHIrr_Wm2"],
        },
        "trk sub atts": (
            [f"TA{n:02d}MtrI_A" for n in range(1, 9)]
            + [f"TA{n:02d}TiltAng_Deg" for n in range(1, 9)]
        ),
        "meter pipoint": "KNSAS.KNS1_S001.PMCAISOBu.Sts.P_kW",
        "MW adjust": -0.001,
    },
    "Kent South": {
        "acmod atts": [f"M{n}.Sts.Running" for n in range(1, 5)],
        "inv atts": [f"DCCT0{n:02d}.Sts.I_A" for n in range(1, 21)],
        "met ids": ["Met3"],
        "metsta atts": ["OE.POA", "PnlTemp_C", "WindSpdAvg_mps", "GHIrr_Wm2"],
        "trk sub atts": (
            [f"TA{n:02d}MtrI_A" for n in range(1, 13)]
            + [f"TA{n:02d}TiltAng_Deg" for n in range(1, 13)]
        ),
        "meter pipoint": "KENTS.KSS1_S001_SL.KSS1_S001.PMCAISOPri.Sts.P_kW",
        "MW adjust": -0.001,
    },
    "Maplewood 1": {
        "metsta atts": {
            "MET009": ["OE.POA", "PO_BOMAvg_degC", "FI_Wthr1WS_ms"],
            "MET022": ["OE.POA", "PO_BOMAvg_degC", "FI_Wthr1WS_ms"],
            "MET044": ["PO_BOMAvg_degC", "FI_Wthr1WS_ms"],
            "MET066": ["OE.POA", "PO_BOMAvg_degC", "FI_Wthr1WS_ms"],
        },
        "meter pipoint": "MPLWD.MPTLOSS_CM_PPA1_RM1_W_CALC_MW",
    },
    "Maplewood 2": {
        "met ids": ["MET082"],
        "metsta atts": ["OE.POA", "FI_Wthr1WS_ms"],  # 'PO_BOMAvg_degC', 'FI_GHI_C'],
        "meter pipoint": "MPLWD.MPTLOSS_CM_PPA2_RM2_W_CALC_MW",
    },
    "Maricopa West": {
        # 'acmod atts': [],
        "met ids": ["Met1", "Met2"],
        "metsta atts": [
            "OE.POA",
            "OE.Module_Temp",
            "OE.Wind_Speed",
            "Global Horizontal Irradiance",
        ],
        "trk sub atts": ["OE.AvgPos", "Tracker GPS Setpoint"],
        "meter pipoint": "MARWE.CAISO Main Meter Real Power-Ph 3 (POD)",
        "MW adjust": 1,
    },
    "MS3": {
        "met ids": ["Met1", "Met3"],
        "metsta atts": ["OE.POA", "GHI"],
        "meter pipoint": "SUMS3.MS3.PLC1.0S_POI01.P",
        "MW adjust": 1,
    },
    "Mulberry": {
        # 'acmod atts': [],
        "inv atts": [f"{n}_MC_GENERIC_CURRENT_DC_ZONE_A" for n in range(1, 9)],
        "met ids": ["A1", "A4", "A6", "B2"],
        "metsta atts": ["OE.POA", "000003_MC_GENERIC_TEMPERATURE_MODULE_C"],
        "meter pipoint": "MLBRY.SEL_735.000004_MC_GENERIC_POWER_ACTIVE_W_W",
        "MW adjust": 0.000001,
    },
    "Old River One": {
        "acmod atts": [f"M{n}.Sts.Running" for n in range(1, 5)],
        "inv atts": [f"DCCT0{n:02d}.Sts.I_A" for n in range(1, 19)],
        "metsta atts": {
            "P002": ["MST001.Sts.Temp_C", "MST002.Sts.Temp_C"],
            "P006": (
                ["OE.POA", "WS001.Sts.WindSpdAvg_mps", "WS001.Sts.GHIrr_Wm2"]
                + [f"MST00{n}.Sts.Temp_C" for n in range(1, 5)]
            ),
        },
        "trk sub atts": (
            [f"TA{n:02d}MtrI_A" for n in range(1, 13)]
            + [f"TA{n:02d}TiltAng_Deg" for n in range(1, 13)]
        ),
        "meter pipoint": "OLRVO.ORS1_S001.PMCAISOPri.Sts.P_kW",
        "MW adjust": -0.001,
    },
    "Pavant": {
        # 'acmod atts': [],
        "inv atts": [f"AMP{n:02d}" for n in range(1, 20)],
        "metsta atts": {
            "Met04": ["OE.POA", "MTS1"],
            "Met13": ["OE.POA"],
            "Met18": ["OE.POA"],
        },
        "trk sub atts": ["OE.AvgPos_TM", "SETPOINT"],
        "meter pipoint": "PAVNT.SS_AI_735_MW",
        "MW adjust": 1,
    },
    "Richland": {
        # 'acmod atts': [],
        "inv atts": [f"CMB{n:02d}_INDCCURR" for n in range(1, 25)],
        "metsta atts": {
            "WS01": ["AIR_TEMP_ACT_C", "WIND_MS_ACT"],
            "WS02": ["OE.POA", "AIR_TEMP_ACT_C", "WIND_MS_ACT"],
        },
        "meter pipoint": "RCH1.RL_METER_KW_TOT",
        "MW adjust": 0.001,
    },
    "Sweetwater": {
        # 'acmod atts': [],
        "inv atts": [f"DC.Current_DC{n}" for n in range(1, 17)],
        "met ids": ["B1.03", "B1.14", "B2.10"],
        "metsta atts": ["OE.POA", "GHI_1", "Wind_Speed", "BOM_Temp_1", "BOM_Temp_2", "BOM_Temp_3"],
        "trk atts": sweetwater_trk_att_dict,
        "meter pipoint": "SWATR.Meter.Rtac.M_H1.kW",
        "MW adjust": 0.001,
    },
    "Three Peaks": {
        # 'acmod atts': [],
        "inv atts": (
            [f"DC1.DCamps{n:02d}" for n in range(1, 11)] + [f"DC2.DCamps{n}" for n in range(11, 20)]
        ),
        "metsta atts": ["OE.POA", "Wind_Speed", "GHI_1"],
        "trk atts": threepeaks_trk_att_dict,
        "meter pipoint": "THRPK.PACMETER.RTAC.PAC.KW",
        "MW adjust": 0.001,
    },
    "West Antelope": {
        # 'acmod atts': [],
        "metsta atts": {
            "Met1": ["OE.POA", "OE.Module_Temp", "OE.Wind_Speed", "Irradiance GHI"],
            "Met2": ["OE.POA", "OE.Wind_Speed", "Irradiance GHI"],
        },
        "trk atts": ["OE.AvgPos", "Tracker GPS Setpoint"],
        "meter pipoint": "TCAAW.Antelope West DNP Client.DNP Client.CAISO Meter MW-3Ph (POD)",
        "MW adjust": 1,
    },
}


solarAFpath = "\\\\CORP-PISQLAF\\Onward Energy\\Renewable Fleet\\Solar Assets"
solar_meter_attPaths = {
    site: os.sep.join([solarAFpath, site, "Meter|OE_MeterMW"]) for site in pqmeta
}


windAFpath = "\\\\CORP-PISQLAF\\Onward Energy\\Renewable Fleet\\Wind Assets"
get_att_path = lambda site, list_: os.sep.join([windAFpath, site, *list_])
wind_meter_atts = {
    "Bingham": ["RTAC", "ISO|ISO_Actual_Generation"],
    "Hancock": ["RTAC", "ISO|ISO_ACTUAL_GENER"],
    "High Sheldon": ["WTG|Sum of Power"],  # not meter
    "Oakfield": ["RTAC", "ISO|ISO_Actual_GEN"],
    "Palouse": ["RTAC", "Avista|AVISTA_JEM_PRI_MW_adj"],
    "Route 66": ["RTAC", "ERCOT|Meter_Adj"],
    "South Plains II": ["RTAC", "SubSt|SEL_735_CMET_Sum"],
    "Sunflower": ["RTAC", "WAPA|POI_MW"],
    "Turkey Track": ["WTG|Sum of Power"],  # not meter,
    "Willow Creek": ["WTG|Sum of Power"],  # not meter
}
wind_meter_attPaths = {site: get_att_path(site, list_) for site, list_ in wind_meter_atts.items()}


def format_piquerydata(df, userTZ, projTZ):
    df.index = pd.to_datetime(df.index.astype(str))
    df.index = df.index.tz_localize(userTZ)  # , ambiguous='infer')
    df.index = df.index.tz_convert(projTZ)
    df.index = df.index.tz_localize(None)
    df.index = df.index.rename("Timestamp")
    df = df.astype(str).apply(pd.to_numeric, errors="coerce")
    return df


def pivot_pidata(df, pivotcolumn):
    df = df.pivot(columns=pivotcolumn)
    df.columns = df.columns.map("_".join)
    return df


def format_ppcdata(df, sitename):
    # input dataframe must have columns 'xcel_MW_setpoint' & 'xcel_MW_cmd_request'
    # add new column 'Curt_SP' & replace setpoint=zero w/ site capacity
    site_capacity = af_dict["SystemSize"]["MWAC"][sitename]
    df["Curt_SP"] = df["xcel_MW_setpoint"]
    df["Curt_SP"] = df["Curt_SP"].mask(df["xcel_MW_cmd_request"] == 0, site_capacity)
    return df


"""NEW REFERENCE CLASS - added 08/30/2023"""


class SolarSite:
    def __init__(
        self,
        inverters,
        inv_atts,
        cmb_type,
        combiners,
        cmb_invatts,
        cmb_atts,
        metstations,
        met_atts,
        trackers,
        trk_sub,
        trk_atts,
        trk_subatts,
        ppc_atts,
        meter_pipoint,
        meter_scaling,
    ):
        self.inverters = inverters
        self.inv_atts = inv_atts
        self.cmb_type = cmb_type
        self.combiners = combiners
        self.cmb_invatts = cmb_invatts  # combiners as inv attributes
        self.cmb_atts = cmb_atts  # combiners as inv subasset attributes
        self.metstations = metstations
        self.met_atts = met_atts
        self.trackers = trackers
        self.trk_sub = trk_sub
        self.trk_atts = trk_atts
        self.trk_subatts = trk_subatts
        self.ppc_atts = ppc_atts
        self.meter_pipoint = meter_pipoint
        self.meter_scaling = meter_scaling


def init_solarsite(sitename):
    site_AFdata = af_dict["AF_Solar_V3"][sitename]
    params = pqmeta[sitename]
    inv_assets = site_AFdata["Inverters"]["Inverters_Assets"]
    inv_IDs = [*inv_assets]

    met_assets = site_AFdata["Met Stations"]["Met Stations_Assets"]
    met_IDs = [*met_assets]  # default all
    if "met ids" in params:
        met_IDs = params["met ids"]

    cmb_type = "Inverter Attributes"  # default
    cmb_IDs = trk_IDs = trksub_IDs = None
    if "inv sub atts" in params:  # if combiners exist as sub assets of inverters
        cmb_type = "Inverter Sub-assets"
        cmb_IDs_ = [*inv_assets[inv_IDs[0]][f"{inv_IDs[0]}_Subassets"]]
        cmb_IDs = [c for c in cmb_IDs_ if "Combiner Boxes" not in c]  # temp, impval restructuring

    if "trk atts" in params and site_AFdata.get("Trackers"):
        trk_assets = site_AFdata["Trackers"]["Trackers_Assets"]
        trk_IDs = [*trk_assets]

    elif "trk sub atts" in params and site_AFdata.get("Trackers"):
        trk_assets = site_AFdata["Trackers"]["Trackers_Assets"]
        trk_IDs = [*trk_assets]
        get_subIDs = lambda trk: [
            *site_AFdata["Trackers"]["Trackers_Assets"][trk][f"{trk}_Subassets"]
        ]
        trksub_IDs = {trk: get_subIDs(trk) for trk in trk_IDs}

    params = {
        "inverters": inv_IDs,
        "inv_atts": "OE.ActivePower",
        "cmb_type": cmb_type,
        "combiners": cmb_IDs,
        "cmb_invatts": params.get("inv atts"),
        "cmb_atts": params.get("inv sub atts"),
        "metstations": met_IDs,
        "met_atts": params.get("metsta atts"),
        "trackers": trk_IDs,
        "trk_sub": trksub_IDs,
        "trk_atts": params.get("trk atts"),
        "trk_subatts": params.get("trk sub atts"),
        "ppc_atts": params.get("ppc atts"),
        "meter_pipoint": params.get("meter pipoint"),
        "meter_scaling": params.get("MW adjust"),
    }
    return SolarSite(**params)


# ref_site_names = [*pqmeta]
# site_object_list = list(map(init_solarsite, ref_site_names))
# solarsites = dict(zip(ref_site_names, site_object_list))
