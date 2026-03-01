-- ============================================================
-- AgentForge RCM — OpenEMR Clinical History Population
-- ============================================================
-- Populates conditions (lists), medications (prescriptions),
-- insurance (insurance_data), and clinical notes (pnotes)
-- for the 12 seeded patients using direct SQL.
--
-- Patient PID map:
--   3  John Smith          P001  Cigna HMO
--   4  Maria Gonzalez      P012  Aetna PPO Gold
--   5  Mary Johnson        P002  UnitedHealth PPO
--   6  Robert Davis        P003  Medicare Part B
--   7  Sarah Chen          P004  Cigna PPO Select
--   8  James Wilson        P005  Aetna HMO
--   9  Emily Rodriguez     P006  Aetna PPO Silver  ← PENICILLIN allergy + Amoxicillin Rx!
--  10  David Kim           P007  Blue Cross PPO
--  11  Patricia Moore      P008  Medicare Advantage
--  12  Alex Turner         P009  UnitedHealth Student
--  13  Maria Santos        P010  Cigna Select
--  14  Thomas Lee          P011  Blue Cross HMO
-- ============================================================

-- ============================================================
-- PART 1: MEDICAL CONDITIONS  (lists table, type='medical_problem')
-- ============================================================

INSERT INTO lists (pid, `date`, `type`, subtype, title, diagnosis, begdate, activity, user, groupname)
VALUES
-- John Smith (pid=3)
(3,  NOW(), 'medical_problem', '', 'Type 2 Diabetes',        'E11',     '2020-01-10', 1, 'admin', 'Default'),
(3,  NOW(), 'medical_problem', '', 'Hypertension',           'I10',     '2020-01-10', 1, 'admin', 'Default'),
(3,  NOW(), 'medical_problem', '', 'Chronic Kidney Disease', 'N18',     '2021-06-15', 1, 'admin', 'Default'),

-- Maria Gonzalez (pid=4)
(4,  NOW(), 'medical_problem', '', 'Malignant Neoplasm Left Breast', 'C50.912', '2025-11-01', 1, 'admin', 'Default'),
(4,  NOW(), 'medical_problem', '', 'Estrogen Receptor Positive Status', 'Z17.0', '2025-11-01', 1, 'admin', 'Default'),
(4,  NOW(), 'medical_problem', '', 'Invasive Ductal Carcinoma Grade 2', 'C50.912', '2025-11-01', 1, 'admin', 'Default'),

-- Mary Johnson (pid=5)
(5,  NOW(), 'medical_problem', '', 'Rheumatoid Arthritis',   'M05',     '2019-08-01', 1, 'admin', 'Default'),
(5,  NOW(), 'medical_problem', '', 'Osteoporosis',           'M81',     '2020-03-15', 1, 'admin', 'Default'),

-- Robert Davis (pid=6)
(6,  NOW(), 'medical_problem', '', 'Atrial Fibrillation',    'I48',     '2018-03-05', 1, 'admin', 'Default'),
(6,  NOW(), 'medical_problem', '', 'Heart Failure',          'I50',     '2019-07-18', 1, 'admin', 'Default'),
(6,  NOW(), 'medical_problem', '', 'Hypertension',           'I10',     '2015-01-01', 1, 'admin', 'Default'),

-- Sarah Chen (pid=7)
(7,  NOW(), 'medical_problem', '', 'Major Depressive Disorder', 'F32',  '2021-03-15', 1, 'admin', 'Default'),
(7,  NOW(), 'medical_problem', '', 'Chronic Pain Syndrome',  'R52',     '2022-01-10', 1, 'admin', 'Default'),
(7,  NOW(), 'medical_problem', '', 'Hypertension',           'I10',     '2021-03-15', 1, 'admin', 'Default'),

-- James Wilson (pid=8)
(8,  NOW(), 'medical_problem', '', 'Type 2 Diabetes',        'E11',     '2018-05-20', 1, 'admin', 'Default'),
(8,  NOW(), 'medical_problem', '', 'Hyperlipidemia',         'E78',     '2018-05-20', 1, 'admin', 'Default'),
(8,  NOW(), 'medical_problem', '', 'Gout',                   'M10',     '2020-02-10', 1, 'admin', 'Default'),

-- Emily Rodriguez (pid=9)
(9,  NOW(), 'medical_problem', '', 'Bacterial Infection',    'A49',     '2026-02-20', 1, 'admin', 'Default'),
(9,  NOW(), 'medical_problem', '', 'Hypertension',           'I10',     '2022-06-15', 1, 'admin', 'Default'),

-- David Kim (pid=10)
(10, NOW(), 'medical_problem', '', 'Bipolar Disorder Type I', 'F31',    '2017-11-01', 1, 'admin', 'Default'),
(10, NOW(), 'medical_problem', '', 'Hypertension',           'I10',     '2019-04-01', 1, 'admin', 'Default'),
(10, NOW(), 'medical_problem', '', 'Hypothyroidism',         'E03',     '2016-04-22', 1, 'admin', 'Default'),

-- Patricia Moore (pid=11)
(11, NOW(), 'medical_problem', '', 'Osteoporosis',           'M81',     '2019-09-10', 1, 'admin', 'Default'),
(11, NOW(), 'medical_problem', '', 'Chronic Kidney Disease', 'N18',     '2020-01-01', 1, 'admin', 'Default'),
(11, NOW(), 'medical_problem', '', 'Hypertension',           'I10',     '2018-06-01', 1, 'admin', 'Default'),

-- Alex Turner (pid=12)
(12, NOW(), 'medical_problem', '', 'Annual Wellness Visit',  'Z00',     '2026-01-14', 1, 'admin', 'Default'),

-- Maria Santos (pid=13)
(13, NOW(), 'medical_problem', '', 'Post-Discharge Monitoring', 'Z09',  '2026-02-01', 1, 'admin', 'Default'),

-- Thomas Lee (pid=14)
(14, NOW(), 'medical_problem', '', 'Preventive Care Examination', 'Z01', '2026-01-30', 1, 'admin', 'Default');


-- ============================================================
-- PART 2: PRESCRIPTIONS
-- ============================================================

INSERT INTO prescriptions (patient_id, `date_added`, drug, dosage, `start_date`, quantity, `active`, provider_id, txDate, usage_category_title, request_intent_title)
VALUES
-- John Smith (pid=3)
(3,  NOW(), 'Metformin',    '500mg twice daily',  '2023-01-10', '30', 1, 1, NOW(), 'community', 'order'),
(3,  NOW(), 'Lisinopril',   '10mg once daily',    '2023-01-10', '30', 1, 1, NOW(), 'community', 'order'),
(3,  NOW(), 'Atorvastatin', '20mg once daily',    '2023-06-15', '30', 1, 1, NOW(), 'community', 'order'),

-- Maria Gonzalez (pid=4) — oncology regimen
(4,  NOW(), 'Palbociclib',  '125mg once daily (21-day cycle)', '2026-02-15', '21', 1, 1, NOW(), 'community', 'order'),
(4,  NOW(), 'Letrozole',    '2.5mg once daily',   '2025-12-01', '30', 1, 1, NOW(), 'community', 'order'),
(4,  NOW(), 'Tamoxifen',    '20mg once daily',    '2024-03-01', '30', 1, 1, NOW(), 'community', 'order'),

-- Mary Johnson (pid=5)
(5,  NOW(), 'Methotrexate', '15mg once weekly',   '2022-08-01', '4',  1, 1, NOW(), 'community', 'order'),
(5,  NOW(), 'Folic Acid',   '1mg once daily',     '2022-08-01', '30', 1, 1, NOW(), 'community', 'order'),
(5,  NOW(), 'Prednisone',   '5mg once daily',     '2023-11-20', '30', 1, 1, NOW(), 'community', 'order'),

-- Robert Davis (pid=6)
(6,  NOW(), 'Warfarin',     '5mg once daily',     '2021-03-05', '30', 1, 1, NOW(), 'community', 'order'),
(6,  NOW(), 'Digoxin',      '0.125mg once daily', '2021-03-05', '30', 1, 1, NOW(), 'community', 'order'),
(6,  NOW(), 'Furosemide',   '40mg once daily',    '2022-07-18', '30', 1, 1, NOW(), 'community', 'order'),

-- Sarah Chen (pid=7)
(7,  NOW(), 'Sertraline',   '100mg once daily',   '2023-03-15', '30', 1, 1, NOW(), 'community', 'order'),
(7,  NOW(), 'Tramadol',     '50mg as needed',     '2024-01-10', '20', 1, 1, NOW(), 'community', 'order'),
(7,  NOW(), 'Lisinopril',   '5mg once daily',     '2023-03-15', '30', 1, 1, NOW(), 'community', 'order'),

-- James Wilson (pid=8)
(8,  NOW(), 'Metformin',    '1000mg twice daily', '2021-05-20', '60', 1, 1, NOW(), 'community', 'order'),
(8,  NOW(), 'Atorvastatin', '40mg once daily',    '2021-05-20', '30', 1, 1, NOW(), 'community', 'order'),
(8,  NOW(), 'Allopurinol',  '300mg once daily',   '2022-02-10', '30', 1, 1, NOW(), 'community', 'order'),
(8,  NOW(), 'Amlodipine',   '5mg once daily',     '2023-08-01', '30', 1, 1, NOW(), 'community', 'order'),

-- Emily Rodriguez (pid=9) — ⚠️ PENICILLIN ALLERGY + AMOXICILLIN = DRUG INTERACTION DEMO
(9,  NOW(), 'Amoxicillin',  '500mg three times daily', '2026-02-20', '21', 1, 1, NOW(), 'community', 'order'),
(9,  NOW(), 'Lisinopril',   '10mg once daily',    '2024-06-15', '30', 1, 1, NOW(), 'community', 'order'),

-- David Kim (pid=10)
(10, NOW(), 'Lithium',      '300mg three times daily', '2020-11-01', '90', 1, 1, NOW(), 'community', 'order'),
(10, NOW(), 'Ibuprofen',    '400mg as needed',    '2025-12-05', '20', 1, 1, NOW(), 'community', 'order'),
(10, NOW(), 'Levothyroxine','75mcg once daily',   '2019-04-22', '30', 1, 1, NOW(), 'community', 'order'),

-- Patricia Moore (pid=11)
(11, NOW(), 'Alendronate',  '70mg once weekly',   '2022-09-10', '4',  1, 1, NOW(), 'community', 'order'),
(11, NOW(), 'Lisinopril',   '5mg once daily',     '2022-09-10', '30', 1, 1, NOW(), 'community', 'order'),
(11, NOW(), 'Calcium Carbonate', '600mg twice daily', '2022-09-10', '60', 1, 1, NOW(), 'community', 'order');


-- ============================================================
-- PART 3: INSURANCE DATA
-- ============================================================

INSERT INTO insurance_data (pid, `type`, provider, plan_name, policy_number, group_number, subscriber_fname, subscriber_lname, subscriber_relationship, `date`, uuid)
VALUES
-- John Smith
(3,  'primary', 'Cigna',        'Cigna HMO Plus',       'CIG-456123', 'GRP-CIG-001', 'John',     'Smith',     'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- Maria Gonzalez
(4,  'primary', 'Aetna',        'Aetna PPO Gold',        'AET-789012', 'GRP-AET-012', 'Maria',    'Gonzalez',  'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- Mary Johnson
(5,  'primary', 'UnitedHealth', 'UHC Choice Plus PPO',   'UHC-321654', 'GRP-UHC-002', 'Mary',     'Johnson',   'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- Robert Davis
(6,  'primary', 'Medicare',     'Medicare Part B',       'MED-654321', 'GRP-MED-003', 'Robert',   'Davis',     'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- Sarah Chen
(7,  'primary', 'Cigna',        'Cigna PPO Select',      'CIG-789456', 'GRP-CIG-004', 'Sarah',    'Chen',      'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- James Wilson
(8,  'primary', 'Aetna',        'Aetna HMO Standard',   'AET-234567', 'GRP-AET-005', 'James',    'Wilson',    'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- Emily Rodriguez
(9,  'primary', 'Aetna',        'Aetna PPO Silver',      'AET-112233', 'GRP-AET-006', 'Emily',    'Rodriguez', 'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- David Kim
(10, 'primary', 'Blue Cross',   'BCBS PPO Classic',      'BCB-345678', 'GRP-BCB-007', 'David',    'Kim',       'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- Patricia Moore
(11, 'primary', 'Medicare',     'Medicare Advantage',    'MED-567890', 'GRP-MED-008', 'Patricia', 'Moore',     'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- Alex Turner
(12, 'primary', 'UnitedHealth', 'UHC Student Health',    'UHC-901234', 'GRP-UHC-009', 'Alex',     'Turner',    'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- Maria Santos
(13, 'primary', 'Cigna',        'Cigna Select Plus',     'CIG-567890', 'GRP-CIG-010', 'Maria',    'Santos',    'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-',''))),
-- Thomas Lee
(14, 'primary', 'Blue Cross',   'BCBS HMO Essential',   'BCB-123456', 'GRP-BCB-011', 'Thomas',   'Lee',       'self', '2024-01-01', UNHEX(REPLACE(UUID(),'-','')));


-- ============================================================
-- PART 4: CLINICAL NOTES (pnotes)
-- ============================================================

INSERT INTO pnotes (pid, `date`, body, title, user, activity, authorized, message_status)
VALUES
(3,  NOW(),
 'Patient: John Smith | DOB: 1965-03-15 | MRN: MRN-P001\nDiagnosis: Type 2 Diabetes (E11), Hypertension (I10), Chronic Kidney Disease Stage 3 (N18).\nAllergies: Penicillin (anaphylaxis), Sulfa (rash).\nCurrent meds: Metformin 500mg BID, Lisinopril 10mg QD, Atorvastatin 20mg QD.\nInsurance: Cigna HMO Plus | CIG-456123.\nNote: Renal function stable. Last HbA1c 7.8%. BP controlled at 128/82.',
 'Clinical Summary — John Smith', 'admin', 1, 1, 'New'),

(4,  NOW(),
 'Patient: Maria J. Gonzalez | DOB: 1978-06-15 | MRN: MRN-00789012\nDiagnosis: Stage II Invasive Ductal Carcinoma left breast (C50.912). ER+/PR+/HER2-.\nAllergies: Penicillin (severe).\nCurrent meds: Palbociclib 125mg QD (21-day cycle), Letrozole 2.5mg QD.\nInsurance: Aetna PPO Gold | AET-789012.\nNote: Ki-67 28%. Prior auth submitted for Palbociclib per Aetna Policy Bulletin #0876. Criteria 4 (ECOG) and 5 (CBC) documentation MISSING — denial risk HIGH.',
 'Clinical Summary — Maria Gonzalez', 'admin', 1, 1, 'New'),

(5,  NOW(),
 'Patient: Mary Johnson | DOB: 1978-07-22\nDiagnosis: Rheumatoid Arthritis (M05), Osteoporosis (M81).\nAllergies: Aspirin (GI upset), Ibuprofen (adverse reaction).\nCurrent meds: Methotrexate 15mg weekly, Folic Acid 1mg QD, Prednisone 5mg QD.\nInsurance: UnitedHealth Choice Plus PPO | UHC-321654.\nNote: Bone density scan due Q6M. Avoid NSAIDs due to documented allergy.',
 'Clinical Summary — Mary Johnson', 'admin', 1, 1, 'New'),

(6,  NOW(),
 'Patient: Robert Davis | DOB: 1952-11-08\nDiagnosis: Atrial Fibrillation (I48), Heart Failure (I50), Hypertension (I10).\nAllergies: None documented.\nCurrent meds: Warfarin 5mg QD (INR target 2.0-3.0), Digoxin 0.125mg QD, Furosemide 40mg QD.\nInsurance: Medicare Part B | MED-654321.\nNote: INR last 2.4. EF 40%. Monitor for signs of decompensation.',
 'Clinical Summary — Robert Davis', 'admin', 1, 1, 'New'),

(7,  NOW(),
 'Patient: Sarah Chen | DOB: 1985-04-12\nDiagnosis: Major Depressive Disorder (F32), Chronic Pain Syndrome (R52), Hypertension (I10).\nAllergies: Codeine (nausea/vomiting), Sulfa (rash).\nCurrent meds: Sertraline 100mg QD, Tramadol 50mg PRN, Lisinopril 5mg QD.\nInsurance: Cigna PPO Select | CIG-789456.\nNote: PHQ-9 score 12 (moderate). Pain management review scheduled.',
 'Clinical Summary — Sarah Chen', 'admin', 1, 1, 'New'),

(8,  NOW(),
 'Patient: James Wilson | DOB: 1962-09-30\nDiagnosis: Type 2 Diabetes (E11), Hyperlipidemia (E78), Gout (M10).\nAllergies: Penicillin (anaphylaxis).\nCurrent meds: Metformin 1000mg BID, Atorvastatin 40mg QD, Allopurinol 300mg QD, Amlodipine 5mg QD.\nInsurance: Aetna HMO Standard | AET-234567.\nNote: LDL 88. Uric acid 6.2. Gout flare-free x12 months.',
 'Clinical Summary — James Wilson', 'admin', 1, 1, 'New'),

(9,  NOW(),
 '⚠️ ALLERGY ALERT — DRUG CONFLICT DETECTED\nPatient: Emily Rodriguez | DOB: 1986-02-14\nDiagnosis: Bacterial Infection (A49), Hypertension (I10).\nAllergies: PENICILLIN (severe anaphylaxis — documented EpiPen prescribed).\nCurrent meds: Amoxicillin 500mg TID ← CONTRAINDICATED (Penicillin-class), Lisinopril 10mg QD.\nInsurance: Aetna PPO Silver | AET-112233.\nNote: CRITICAL — Amoxicillin is a penicillin-class antibiotic. Patient has documented severe Penicillin allergy. Prescriber must be notified immediately. Consider Azithromycin or Doxycycline as alternative.',
 'ALLERGY ALERT — Emily Rodriguez', 'admin', 1, 1, 'New'),

(10, NOW(),
 'Patient: David Kim | DOB: 1969-07-05\nDiagnosis: Bipolar Disorder Type I (F31), Hypertension (I10), Hypothyroidism (E03).\nAllergies: None documented.\nCurrent meds: Lithium 300mg TID (serum level 0.8 mEq/L), Ibuprofen 400mg PRN, Levothyroxine 75mcg QD.\nInsurance: BCBS PPO Classic | BCB-345678.\nNote: CAUTION — Ibuprofen may increase Lithium levels (interaction risk). Monitor serum lithium q3M. TSH 2.1.',
 'Clinical Summary — David Kim', 'admin', 1, 1, 'New'),

(11, NOW(),
 'Patient: Patricia Moore | DOB: 1954-12-20\nDiagnosis: Osteoporosis (M81), Chronic Kidney Disease (N18), Hypertension (I10).\nAllergies: Aspirin (GI bleed history), Codeine (respiratory depression).\nCurrent meds: Alendronate 70mg weekly, Lisinopril 5mg QD, Calcium Carbonate 600mg BID.\nInsurance: Medicare Advantage | MED-567890.\nNote: eGFR 42. DEXA T-score -2.8. Fall risk assessment completed.',
 'Clinical Summary — Patricia Moore', 'admin', 1, 1, 'New'),

(12, NOW(),
 'Patient: Alex Turner | DOB: 1996-08-14\nNo chronic conditions. Annual wellness visit.\nAllergies: None documented.\nInsurance: UnitedHealth Student Health | UHC-901234.\nNote: All preventive screenings up to date. BMI 23.4. No medications.',
 'Clinical Summary — Alex Turner', 'admin', 1, 1, 'New'),

(13, NOW(),
 'Patient: Maria Santos | DOB: 1989-03-22\nPost-discharge monitoring follow-up.\nAllergies: None documented.\nInsurance: Cigna Select Plus | CIG-567890.\nNote: Discharged 2026-01-28. Vital signs stable. Follow-up labs pending.',
 'Clinical Summary — Maria Santos', 'admin', 1, 1, 'New'),

(14, NOW(),
 'Patient: Thomas Lee | DOB: 1979-11-30\nPreventive care examination.\nAllergies: None documented.\nInsurance: BCBS HMO Essential | BCB-123456.\nNote: Routine annual physical. Cholesterol panel ordered. BP 118/76.',
 'Clinical Summary — Thomas Lee', 'admin', 1, 1, 'New');


-- ============================================================
-- VERIFICATION
-- ============================================================
SELECT 'CONDITIONS' AS table_name, COUNT(*) AS rows_inserted FROM lists WHERE pid BETWEEN 3 AND 14 AND type='medical_problem'
UNION ALL
SELECT 'PRESCRIPTIONS', COUNT(*) FROM prescriptions WHERE patient_id BETWEEN 3 AND 14
UNION ALL
SELECT 'INSURANCE', COUNT(*) FROM insurance_data WHERE pid BETWEEN 3 AND 14
UNION ALL
SELECT 'CLINICAL NOTES', COUNT(*) FROM pnotes WHERE pid BETWEEN 3 AND 14;
