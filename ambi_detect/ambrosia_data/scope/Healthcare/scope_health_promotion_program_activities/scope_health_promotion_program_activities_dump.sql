BEGIN TRANSACTION;
CREATE TABLE Activities (
    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_name TEXT,
    description TEXT
);
INSERT INTO "Activities" VALUES(1,'Exercise Classes','Group exercise sessions that promote physical fitness.');
INSERT INTO "Activities" VALUES(2,'Nutrition Workshops','Educational workshops on healthy eating habits.');
INSERT INTO "Activities" VALUES(3,'Stress Management Sessions','Sessions designed to help participants cope with stress.');
INSERT INTO "Activities" VALUES(4,'Support Group Meetings','Meetings where participants can share experiences and offer support.');
INSERT INTO "Activities" VALUES(5,'Yoga Sessions','Relaxing yoga classes to improve flexibility and reduce stress.');
CREATE TABLE Health_Promotion_Program (
    program_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);
INSERT INTO "Health_Promotion_Program" VALUES(1,'Weight Management Program','This program helps people manage their weight.');
INSERT INTO "Health_Promotion_Program" VALUES(2,'Diabetes Prevention Program','This program educates individuals about diabetes prevention.');
INSERT INTO "Health_Promotion_Program" VALUES(3,'Smoking Cessation Program','This program provides support for quitting smoking.');
INSERT INTO "Health_Promotion_Program" VALUES(4,'Mental Wellness Program','This program offers resources for mental health.');
INSERT INTO "Health_Promotion_Program" VALUES(5,'Heart Health Program','This program focuses on heart disease prevention and management.');
CREATE TABLE Health_Promotion_Program_Activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_id INTEGER,
    activity_id INTEGER,
    FOREIGN KEY(program_id) REFERENCES Health_Promotion_Program(program_id),
    FOREIGN KEY(activity_id) REFERENCES Activities(activity_id)
);
INSERT INTO "Health_Promotion_Program_Activities" VALUES(1,1,1);
INSERT INTO "Health_Promotion_Program_Activities" VALUES(2,1,2);
INSERT INTO "Health_Promotion_Program_Activities" VALUES(3,2,1);
INSERT INTO "Health_Promotion_Program_Activities" VALUES(4,2,2);
INSERT INTO "Health_Promotion_Program_Activities" VALUES(5,3,1);
INSERT INTO "Health_Promotion_Program_Activities" VALUES(6,4,1);
INSERT INTO "Health_Promotion_Program_Activities" VALUES(7,5,1);
CREATE TABLE Participants (
    participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number VARCHAR(15)
);
INSERT INTO "Participants" VALUES(1,'John','Doe','john.doe@example.com','+123-456-7890');
INSERT INTO "Participants" VALUES(2,'Jane','Smith','jane.smith@example.com','+987-654-3210');
INSERT INTO "Participants" VALUES(3,'Alice','Brown','alice.brown@example.com','+555-123-4567');
INSERT INTO "Participants" VALUES(4,'Bob','White','bob.white@example.com','+333-999-8887');
INSERT INTO "Participants" VALUES(5,'Emma','Black','emma.black@example.com','+777-666-5555');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Health_Promotion_Program',5);
INSERT INTO "sqlite_sequence" VALUES('Activities',5);
INSERT INTO "sqlite_sequence" VALUES('Participants',5);
INSERT INTO "sqlite_sequence" VALUES('Health_Promotion_Program_Activities',7);
COMMIT;
