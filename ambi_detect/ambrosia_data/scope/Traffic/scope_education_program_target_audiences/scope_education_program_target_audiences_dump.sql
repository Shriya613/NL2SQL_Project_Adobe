BEGIN TRANSACTION;
CREATE TABLE Education_Program (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);
INSERT INTO "Education_Program" VALUES(1,'Driving Basics','This program covers the basic driving rules');
INSERT INTO "Education_Program" VALUES(2,'Advanced Driving Techniques','Learn advanced techniques such as defensive driving');
INSERT INTO "Education_Program" VALUES(3,'Traffic Laws & Regulations','Understand traffic laws and regulations');
INSERT INTO "Education_Program" VALUES(4,'Accident Prevention','Learn how to prevent accidents on road');
INSERT INTO "Education_Program" VALUES(5,'Emergency Handling','Handling emergencies while driving');
CREATE TABLE Education_Program_Target_Audiences (
    education_program_id INTEGER,
    target_audience_id INTEGER,
    FOREIGN KEY (education_program_id) REFERENCES Education_Program(id),
    FOREIGN KEY (target_audience_id) REFERENCES Target_Audiences(id),
    PRIMARY KEY (education_program_id, target_audience_id)
);
INSERT INTO "Education_Program_Target_Audiences" VALUES(1,2);
INSERT INTO "Education_Program_Target_Audiences" VALUES(2,2);
INSERT INTO "Education_Program_Target_Audiences" VALUES(3,2);
INSERT INTO "Education_Program_Target_Audiences" VALUES(4,2);
INSERT INTO "Education_Program_Target_Audiences" VALUES(1,1);
INSERT INTO "Education_Program_Target_Audiences" VALUES(5,2);
CREATE TABLE Target_Audiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audience_name TEXT
);
INSERT INTO "Target_Audiences" VALUES(1,'Teen Driver Education');
INSERT INTO "Target_Audiences" VALUES(2,'Adult Driver Refresher Course');
INSERT INTO "Target_Audiences" VALUES(3,'Defensive Driving Training');
INSERT INTO "Target_Audiences" VALUES(4,'Road Safety Awareness Program');
INSERT INTO "Target_Audiences" VALUES(5,'Parent-Child Driving Course');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Education_Program',5);
INSERT INTO "sqlite_sequence" VALUES('Target_Audiences',5);
COMMIT;
