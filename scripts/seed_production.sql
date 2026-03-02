-- ============================================================
-- AgentForge RCM — Production Data Seed (v3 — schema-corrected)
-- ============================================================

SET FOREIGN_KEY_CHECKS = 0;

-- ============================================================
-- 0. CLEANUP — remove any previous partial seed
-- ============================================================
DELETE FROM prescriptions   WHERE patient_id BETWEEN 1 AND 12;
DELETE FROM insurance_data  WHERE pid        BETWEEN 1 AND 12;
DELETE FROM form_encounter  WHERE pid        BETWEEN 1 AND 12;
DELETE FROM lists           WHERE pid        BETWEEN 1 AND 12;
DELETE FROM patient_data    WHERE pid        BETWEEN 1 AND 12;

-- ============================================================
-- 1. PATIENTS  (pid explicit — UNIQUE, not auto-increment)
-- ============================================================
INSERT INTO patient_data
  (pid, pubpid, fname, lname, DOB, sex, status, date, regdate, city, state, country_code, language, financial)
VALUES
  (1, 'P001','John',     'Smith',     '1965-03-15','Male',  'active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (2, 'P002','Mary',     'Johnson',   '1978-07-22','Female','active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (3, 'P003','Robert',   'Davis',     '1952-11-08','Male',  'active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (4, 'P004','Sarah',    'Chen',      '1985-04-12','Female','active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (5, 'P005','James',    'Wilson',    '1962-09-30','Male',  'active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (6, 'P006','Emily',    'Rodriguez', '1986-02-14','Female','active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (7, 'P007','David',    'Kim',       '1969-07-05','Male',  'active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (8, 'P008','Patricia', 'Moore',     '1954-12-20','Female','active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (9, 'P009','Alex',     'Turner',    '1996-08-14','Male',  'active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (10,'P010','Maria',    'Santos',    '1989-03-22','Female','active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (11,'P011','Thomas',   'Lee',       '1979-11-30','Male',  'active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English',''),
  (12,'P012','Maria J.', 'Gonzalez',  '1968-05-17','Female','active','2024-01-10','2024-01-10','Philadelphia','PA','USA','English','');

SET @p001=1; SET @p002=2; SET @p003=3; SET @p004=4;
SET @p005=5; SET @p006=6; SET @p007=7; SET @p008=8;
SET @p009=9; SET @p010=10; SET @p011=11; SET @p012=12;

-- ============================================================
-- 2. ALLERGIES  (lists — subtype replaces allergy_type)
-- ============================================================
INSERT INTO lists (pid, type, subtype, title, reaction, severity_al, date, activity) VALUES
  (@p001,'allergy','drug','Penicillin',   'Anaphylaxis','severe',  '2024-01-10',1),
  (@p001,'allergy','drug','Sulfa',        'Rash',       'moderate','2024-01-10',1),
  (@p002,'allergy','drug','Aspirin',      'GI Bleed',   'severe',  '2024-01-10',1),
  (@p002,'allergy','drug','Ibuprofen',    'GI Distress','moderate','2024-01-10',1),
  (@p004,'allergy','drug','Codeine',      'Nausea',     'moderate','2024-01-10',1),
  (@p004,'allergy','drug','Sulfa',        'Rash',       'moderate','2024-01-10',1),
  (@p005,'allergy','drug','Penicillin',   'Hives',      'moderate','2024-01-10',1),
  (@p006,'allergy','drug','Penicillin',   'Anaphylaxis','severe',  '2024-01-10',1),
  (@p008,'allergy','drug','Aspirin',      'GI Bleed',   'severe',  '2024-01-10',1),
  (@p008,'allergy','drug','Codeine',      'Nausea',     'moderate','2024-01-10',1),
  (@p012,'allergy','drug','Penicillin',   'Rash',       'moderate','2024-01-10',1);

-- ============================================================
-- 3. CONDITIONS  (lists type='medical_problem')
-- ============================================================
INSERT INTO lists (pid, type, subtype, title, diagnosis, date, activity) VALUES
  (@p001,'medical_problem','','Type 2 Diabetes',          'E11.9', '2024-01-10',1),
  (@p001,'medical_problem','','Hypertension',              'I10',   '2024-01-10',1),
  (@p001,'medical_problem','','Chronic Kidney Disease',    'N18.3', '2024-01-10',1),
  (@p002,'medical_problem','','Rheumatoid Arthritis',      'M06.9', '2024-01-10',1),
  (@p002,'medical_problem','','Osteoporosis',              'M81.0', '2024-01-10',1),
  (@p003,'medical_problem','','Atrial Fibrillation',       'I48.91','2024-01-10',1),
  (@p003,'medical_problem','','Heart Failure',             'I50.9', '2024-01-10',1),
  (@p003,'medical_problem','','Hypertension',              'I10',   '2024-01-10',1),
  (@p004,'medical_problem','','Depression',                'F32.9', '2024-01-10',1),
  (@p004,'medical_problem','','Chronic Pain',              'G89.29','2024-01-10',1),
  (@p004,'medical_problem','','Hypertension',              'I10',   '2024-01-10',1),
  (@p005,'medical_problem','','Type 2 Diabetes',          'E11.9', '2024-01-10',1),
  (@p005,'medical_problem','','Hyperlipidemia',            'E78.5', '2024-01-10',1),
  (@p005,'medical_problem','','Gout',                      'M10.9', '2024-01-10',1),
  (@p006,'medical_problem','','Bacterial Infection',       'A49.9', '2024-01-10',1),
  (@p006,'medical_problem','','Hypertension',              'I10',   '2024-01-10',1),
  (@p007,'medical_problem','','Bipolar Disorder',          'F31.9', '2024-01-10',1),
  (@p007,'medical_problem','','Hypertension',              'I10',   '2024-01-10',1),
  (@p007,'medical_problem','','Hypothyroidism',            'E03.9', '2024-01-10',1),
  (@p008,'medical_problem','','Osteoporosis',              'M81.0', '2024-01-10',1),
  (@p008,'medical_problem','','Chronic Kidney Disease',    'N18.3', '2024-01-10',1),
  (@p008,'medical_problem','','Hypertension',              'I10',   '2024-01-10',1),
  (@p009,'medical_problem','','Annual Checkup',            'Z00.00','2024-01-10',1),
  (@p010,'medical_problem','','Post-Discharge Monitoring', 'Z09',   '2024-01-10',1),
  (@p011,'medical_problem','','Preventive Care',           'Z00.00','2024-01-10',1),
  (@p012,'medical_problem','','Breast Cancer',             'C50.911','2024-01-10',1),
  (@p012,'medical_problem','','Invasive Ductal Carcinoma', 'C50.911','2024-01-10',1),
  (@p012,'medical_problem','','Post-Surgical Monitoring',  'Z09',   '2024-01-10',1);

-- ============================================================
-- 4. ENCOUNTERS
-- ============================================================
INSERT INTO form_encounter (pid, date, reason, facility_id, onset_date, sensitivity, billing_note) VALUES
  (@p001,'2024-01-15','Diabetes & Hypertension Follow-up',0,'2024-01-15','normal',''),
  (@p002,'2024-01-16','Rheumatoid Arthritis Management',  0,'2024-01-16','normal',''),
  (@p003,'2024-01-17','Cardiac Follow-up',                0,'2024-01-17','normal',''),
  (@p004,'2024-01-18','Mental Health & Pain Management',  0,'2024-01-18','normal',''),
  (@p005,'2024-01-19','Diabetes & Gout Follow-up',        0,'2024-01-19','normal',''),
  (@p006,'2024-01-20','Infection Treatment',              0,'2024-01-20','normal',''),
  (@p007,'2024-01-21','Psychiatric Follow-up',            0,'2024-01-21','normal',''),
  (@p008,'2024-01-22','Osteoporosis Management',          0,'2024-01-22','normal',''),
  (@p009,'2024-01-23','Annual Wellness Visit',            0,'2024-01-23','normal',''),
  (@p010,'2024-01-24','Post-Discharge Follow-up',         0,'2024-01-24','normal',''),
  (@p011,'2024-01-25','Preventive Care Visit',            0,'2024-01-25','normal',''),
  (@p012,'2024-01-26','Oncology Follow-up',               0,'2024-01-26','normal','');

-- ============================================================
-- 5. MEDICATIONS  (prescriptions — unit/interval are int, use NULL)
-- Drug interactions seeded intentionally:
--   P002: Methotrexate + Prednisone (immunosuppression HIGH)
--         Ibuprofen allergy + Ibuprofen prescribed = DRUG_INTERACTION_HIGH
--   P003: Warfarin + Digoxin (HIGH interaction)
--   P007: Lithium + Ibuprofen (HIGH — reduces lithium clearance)
-- ============================================================
INSERT INTO prescriptions
  (patient_id, date_added, drug, rxnorm_drugcode, dosage, quantity, route, refills, active, txDate, usage_category_title, request_intent_title, note)
VALUES
  -- P001 John Smith
  (@p001,'2024-01-15','Metformin',  '860975','500mg', '60', 'Oral',3,1,'2024-01-15','','','twice daily for Type 2 Diabetes'),
  (@p001,'2024-01-15','Lisinopril', '203644','10mg',  '30', 'Oral',3,1,'2024-01-15','','','once daily for Hypertension and CKD'),
  (@p001,'2024-01-15','Atorvastatin','301542','20mg', '30', 'Oral',3,1,'2024-01-15','','','once daily for cholesterol'),

  -- P002 Mary Johnson (HIGH RISK: Methotrexate + Ibuprofen allergy + Prednisone)
  (@p002,'2024-01-16','Methotrexate','105586','15mg', '4',  'Oral',3,1,'2024-01-16','','','weekly for Rheumatoid Arthritis — monitor LFTs'),
  (@p002,'2024-01-16','Folic Acid',  '4495',  '1mg',  '30', 'Oral',3,1,'2024-01-16','','','once daily to reduce methotrexate side effects'),
  (@p002,'2024-01-16','Prednisone',  '763112','5mg',  '30', 'Oral',3,1,'2024-01-16','','','once daily for RA flare — immunosuppression risk with MTX'),

  -- P003 Robert Davis (HIGH RISK: Warfarin + Digoxin)
  (@p003,'2024-01-17','Warfarin',   '202421','5mg',   '30', 'Oral',3,1,'2024-01-17','','','once daily for Atrial Fibrillation — INR monitoring required'),
  (@p003,'2024-01-17','Digoxin',    '197604','0.125mg','30','Oral',3,1,'2024-01-17','','','once daily for rate control — HIGH interaction with Warfarin'),
  (@p003,'2024-01-17','Furosemide', '202991','40mg',  '30', 'Oral',3,1,'2024-01-17','','','once daily for Heart Failure'),

  -- P004 Sarah Chen
  (@p004,'2024-01-18','Sertraline', '36437','50mg',   '30', 'Oral',3,1,'2024-01-18','','','once daily for Depression'),
  (@p004,'2024-01-18','Tramadol',   '41493','50mg',   '30', 'Oral',1,1,'2024-01-18','','','as needed for Chronic Pain — serotonin risk with Sertraline'),
  (@p004,'2024-01-18','Lisinopril', '203644','10mg',  '30', 'Oral',3,1,'2024-01-18','','','once daily for Hypertension'),

  -- P005 James Wilson
  (@p005,'2024-01-19','Metformin',  '860975','1000mg','60', 'Oral',3,1,'2024-01-19','','','twice daily for Type 2 Diabetes'),
  (@p005,'2024-01-19','Atorvastatin','301542','40mg', '30', 'Oral',3,1,'2024-01-19','','','once daily for Hyperlipidemia'),
  (@p005,'2024-01-19','Allopurinol','202326','300mg', '30', 'Oral',3,1,'2024-01-19','','','once daily for Gout'),
  (@p005,'2024-01-19','Amlodipine', '197361','5mg',   '30', 'Oral',3,1,'2024-01-19','','','once daily for Hypertension'),

  -- P006 Emily Rodriguez (ALERT: Penicillin allergy + Amoxicillin prescribed)
  (@p006,'2024-01-20','Amoxicillin','723',   '500mg', '21', 'Oral',0,1,'2024-01-20','','','three times daily for Bacterial Infection — ALLERGY ALERT: Penicillin class'),
  (@p006,'2024-01-20','Lisinopril', '203644','5mg',   '30', 'Oral',3,1,'2024-01-20','','','once daily for Hypertension'),

  -- P007 David Kim (HIGH RISK: Lithium + Ibuprofen)
  (@p007,'2024-01-21','Lithium',    '19593', '300mg', '90', 'Oral',3,1,'2024-01-21','','','three times daily for Bipolar Disorder — monitor serum levels'),
  (@p007,'2024-01-21','Ibuprofen',  '197805','400mg', '30', 'Oral',1,1,'2024-01-21','','','as needed — HIGH RISK: reduces lithium clearance, toxicity risk'),
  (@p007,'2024-01-21','Levothyroxine','10582','75mcg','30', 'Oral',3,1,'2024-01-21','','','once daily for Hypothyroidism'),

  -- P008 Patricia Moore
  (@p008,'2024-01-22','Alendronate',     '17549','70mg',   '4',  'Oral',3,1,'2024-01-22','','','weekly for Osteoporosis'),
  (@p008,'2024-01-22','Lisinopril',      '203644','10mg',  '30', 'Oral',3,1,'2024-01-22','','','once daily for Hypertension and CKD'),
  (@p008,'2024-01-22','Calcium Carbonate','41257','1000mg','60', 'Oral',3,1,'2024-01-22','','','twice daily for Osteoporosis support'),

  -- P012 Maria Gonzalez (Aetna CPB #0876 prior auth scenario)
  (@p012,'2024-01-26','Palbociclib','1873983','125mg','21','Oral',3,1,'2024-01-26','','','21 days on / 7 days off — CDK4/6 inhibitor for Breast Cancer — prior auth required'),
  (@p012,'2024-01-26','Letrozole',  '72512',  '2.5mg','30','Oral',3,1,'2024-01-26','','','once daily — aromatase inhibitor for ER+ Breast Cancer');

-- ============================================================
-- 6. INSURANCE  (provider is varchar — no FK to insurance_companies)
-- ============================================================
INSERT INTO insurance_data
  (pid, type, provider, plan_name, policy_number, group_number, subscriber_fname, subscriber_lname, subscriber_DOB, policy_type, date)
VALUES
  (@p001,'primary','Aetna',              'Aetna Choice POS II',     'AET-001-2024','GRP-4521','John',     'Smith',    '1965-03-15','',CURDATE()),
  (@p002,'primary','Cigna',              'Cigna Open Access Plus',  'CIG-002-2024','GRP-7832','Mary',     'Johnson',  '1978-07-22','',CURDATE()),
  (@p003,'primary','Medicare',           'Medicare Part B',         'MED-003-2024','GRP-0000','Robert',   'Davis',    '1952-11-08','',CURDATE()),
  (@p004,'primary','Blue Cross',         'BCBS PPO',                'BCB-004-2024','GRP-3310','Sarah',    'Chen',     '1985-04-12','',CURDATE()),
  (@p005,'primary','United Healthcare',  'UHC Choice Plus',         'UHC-005-2024','GRP-6621','James',    'Wilson',   '1962-09-30','',CURDATE()),
  (@p006,'primary','Medicaid',           'Medicaid Managed Care',   'MCD-006-2024','GRP-0000','Emily',    'Rodriguez','1986-02-14','',CURDATE()),
  (@p007,'primary','Blue Cross',         'BCBS HMO',                'BCB-007-2024','GRP-3311','David',    'Kim',      '1969-07-05','',CURDATE()),
  (@p008,'primary','Medicare',           'Medicare Advantage',      'MED-008-2024','GRP-0001','Patricia', 'Moore',    '1954-12-20','',CURDATE()),
  (@p009,'primary','United Healthcare',  'UHC Student Health',      'UHC-009-2024','GRP-9901','Alex',     'Turner',   '1996-08-14','',CURDATE()),
  (@p010,'primary','Cigna',             'Cigna PPO',               'CIG-010-2024','GRP-7833','Maria',    'Santos',   '1989-03-22','',CURDATE()),
  (@p011,'primary','Aetna',             'Aetna HMO',               'AET-011-2024','GRP-4522','Thomas',   'Lee',      '1979-11-30','',CURDATE()),
  (@p012,'primary','Aetna',             'Aetna CPB #0876',         'AET-012-2024','GRP-4523','Maria J.', 'Gonzalez', '1968-05-17','',CURDATE());

SET FOREIGN_KEY_CHECKS = 1;

-- ============================================================
-- VERIFY
-- ============================================================
SELECT 'Patients'    AS entity, COUNT(*) AS count FROM patient_data WHERE pubpid LIKE 'P0%'
UNION ALL
SELECT 'Allergies',   COUNT(*) FROM lists WHERE type='allergy'
UNION ALL
SELECT 'Conditions',  COUNT(*) FROM lists WHERE type='medical_problem'
UNION ALL
SELECT 'Medications', COUNT(*) FROM prescriptions WHERE active=1
UNION ALL
SELECT 'Encounters',  COUNT(*) FROM form_encounter WHERE pid BETWEEN 1 AND 12
UNION ALL
SELECT 'Insurance',   COUNT(*) FROM insurance_data WHERE pid BETWEEN 1 AND 12;
