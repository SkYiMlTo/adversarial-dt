(* ===========================================================================
   IEC 61131-3 Structured Text: Two-Tank Level Control
   ===========================================================================
   This program implements hysteresis-based pump control for a two-tank
   water distribution system. It is functionally equivalent to the Python
   implementation in main.py and can be loaded into OpenPLC Runtime v3
   for validation on real PLC hardware.

   Inputs (from Modbus holding registers):
       L1_RAW : INT  — Main tank level (0-65535 → 0.0-5.0 m)
       L2_RAW : INT  — Secondary tank level (0-65535 → 0.0-4.0 m)

   Outputs (to Modbus coil registers):
       PUMP_CMD   : BOOL — Pump enable (TRUE/FALSE)
       VALVE_POS  : INT  — Valve position (0-100%)
   =========================================================================== *)

PROGRAM TwoTankControl
VAR
    (* Inputs *)
    L1_RAW      : INT;
    L2_RAW      : INT;

    (* Outputs *)
    PUMP_CMD    : BOOL := TRUE;
    VALVE_POS   : INT := 60;

    (* Internal *)
    L1          : REAL;
    L2          : REAL;

    (* Thresholds *)
    L1_LOW      : REAL := 1.0;    (* Pump ON threshold [m] *)
    L1_HIGH     : REAL := 4.0;    (* Pump OFF threshold [m] *)
    L1_MAX      : REAL := 5.0;    (* Full scale for L1 *)
    L2_MAX      : REAL := 4.0;    (* Full scale for L2 *)
END_VAR

(* Scale raw register values to engineering units *)
L1 := INT_TO_REAL(L1_RAW) / 65535.0 * L1_MAX;
L2 := INT_TO_REAL(L2_RAW) / 65535.0 * L2_MAX;

(* Hysteresis pump control *)
IF L1 < L1_LOW THEN
    PUMP_CMD := TRUE;
ELSIF L1 > L1_HIGH THEN
    PUMP_CMD := FALSE;
END_IF;

(* Valve position: constant at 60% *)
VALVE_POS := 60;

END_PROGRAM
