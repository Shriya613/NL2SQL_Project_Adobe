BEGIN TRANSACTION;
CREATE TABLE Diagnoses(
    diagnosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id INTEGER,
    disease_id INTEGER,
    date_of_diagnosis DATE,
    FOREIGN KEY (participant_id) REFERENCES Study_Participants(participant_id),
    FOREIGN KEY (disease_id) REFERENCES Diseases(disease_id)
);
INSERT INTO "Diagnoses" VALUES(1,1,1,'2020-04-01');
INSERT INTO "Diagnoses" VALUES(2,1,2,'2020-04-15');
INSERT INTO "Diagnoses" VALUES(3,2,3,'2020-05-01');
INSERT INTO "Diagnoses" VALUES(4,2,4,'2020-05-15');
INSERT INTO "Diagnoses" VALUES(5,3,5,'2020-06-01');
CREATE TABLE Diseases(
    disease_id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_name TEXT
);
INSERT INTO "Diseases" VALUES(1,'Covid-19');
INSERT INTO "Diseases" VALUES(2,'Influenza');
INSERT INTO "Diseases" VALUES(3,'Type II Diabetes');
INSERT INTO "Diseases" VALUES(4,'Coronary Heart Disease');
INSERT INTO "Diseases" VALUES(5,'Obesity');
CREATE TABLE Epidemiological_Study(
    study_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    start_date DATE,
    end_date DATE
);
INSERT INTO "Epidemiological_Study" VALUES(1,'COVID-19 Study','2020-03-01','2021-06-01');
INSERT INTO "Epidemiological_Study" VALUES(2,'Flu Study','2020-08-01','2021-07-01');
INSERT INTO "Epidemiological_Study" VALUES(3,'Diabetes Study','2020-04-01','2021-09-01');
INSERT INTO "Epidemiological_Study" VALUES(4,'Heart Disease Study','2020-05-01','2021-10-01');
INSERT INTO "Epidemiological_Study" VALUES(5,'Obesity Study','2020-06-01','2021-11-01');
CREATE TABLE Epidemiological_Study_Methods(
    epidemiological_study_id INTEGER,
    method_id INTEGER,
    FOREIGN KEY (epidemiological_study_id) REFERENCES Epidemiological_Study(study_id),
    FOREIGN KEY (method_id) REFERENCES Methods(method_id)
);
INSERT INTO "Epidemiological_Study_Methods" VALUES(1,1);
INSERT INTO "Epidemiological_Study_Methods" VALUES(1,2);
INSERT INTO "Epidemiological_Study_Methods" VALUES(2,3);
INSERT INTO "Epidemiological_Study_Methods" VALUES(2,4);
INSERT INTO "Epidemiological_Study_Methods" VALUES(3,5);
INSERT INTO "Epidemiological_Study_Methods" VALUES(2,1);
INSERT INTO "Epidemiological_Study_Methods" VALUES(3,1);
INSERT INTO "Epidemiological_Study_Methods" VALUES(4,1);
INSERT INTO "Epidemiological_Study_Methods" VALUES(5,1);
CREATE TABLE Methods(
    method_id INTEGER PRIMARY KEY AUTOINCREMENT,
    method_name TEXT
);
INSERT INTO "Methods" VALUES(1,'Survey');
INSERT INTO "Methods" VALUES(2,'Blood Test');
INSERT INTO "Methods" VALUES(3,'Questionnaire');
INSERT INTO "Methods" VALUES(4,'Physical Exam');
INSERT INTO "Methods" VALUES(5,'Imaging');
CREATE TABLE Study_Participants(
    participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    epidemiological_study_id INTEGER,
    first_name TEXT,
    last_name TEXT,
    age INT,
    FOREIGN KEY (epidemiological_study_id) REFERENCES Epidemiological_Study(study_id)
);
INSERT INTO "Study_Participants" VALUES(1,1,'John','Doe',30);
INSERT INTO "Study_Participants" VALUES(2,1,'Jane','Smith',25);
INSERT INTO "Study_Participants" VALUES(3,2,'Bob','Brown',40);
INSERT INTO "Study_Participants" VALUES(4,2,'Alice','Green',35);
INSERT INTO "Study_Participants" VALUES(5,3,'Tom','White',50);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Epidemiological_Study',5);
INSERT INTO "sqlite_sequence" VALUES('Methods',5);
INSERT INTO "sqlite_sequence" VALUES('Study_Participants',5);
INSERT INTO "sqlite_sequence" VALUES('Diseases',5);
INSERT INTO "sqlite_sequence" VALUES('Diagnoses',5);
COMMIT;
