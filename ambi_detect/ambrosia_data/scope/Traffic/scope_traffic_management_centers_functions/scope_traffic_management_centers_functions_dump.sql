BEGIN TRANSACTION;
CREATE TABLE Functions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    functionName TEXT);
INSERT INTO "Functions" VALUES(1,'Incident Detection System');
INSERT INTO "Functions" VALUES(2,'Road Closure Notification System');
INSERT INTO "Functions" VALUES(3,'Accident Reporting System');
INSERT INTO "Functions" VALUES(4,'Real-Time Traffic Analysis System');
INSERT INTO "Functions" VALUES(5,'Weather Impact Assessment System');
CREATE TABLE Traffic_Management_Centers(
    centerID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Traffic_Management_Centers" VALUES(1,'TMC1','Los Angeles');
INSERT INTO "Traffic_Management_Centers" VALUES(2,'TMC2','Chicago');
INSERT INTO "Traffic_Management_Centers" VALUES(3,'TMC3','New York City');
INSERT INTO "Traffic_Management_Centers" VALUES(4,'TMC4','Miami');
INSERT INTO "Traffic_Management_Centers" VALUES(5,'TMC5','Houston');
CREATE TABLE Traffic_Management_Centers_Functions(
    TMC_id INTEGER REFERENCES Traffic_Management_Centers(centerID),
    Function_id INTEGER REFERENCES Functions(id));
INSERT INTO "Traffic_Management_Centers_Functions" VALUES(1,1);
INSERT INTO "Traffic_Management_Centers_Functions" VALUES(1,2);
INSERT INTO "Traffic_Management_Centers_Functions" VALUES(2,3);
INSERT INTO "Traffic_Management_Centers_Functions" VALUES(2,4);
INSERT INTO "Traffic_Management_Centers_Functions" VALUES(3,5);
INSERT INTO "Traffic_Management_Centers_Functions" VALUES(2,1);
INSERT INTO "Traffic_Management_Centers_Functions" VALUES(3,1);
INSERT INTO "Traffic_Management_Centers_Functions" VALUES(4,1);
INSERT INTO "Traffic_Management_Centers_Functions" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Functions',5);
INSERT INTO "sqlite_sequence" VALUES('Traffic_Management_Centers',5);
COMMIT;
