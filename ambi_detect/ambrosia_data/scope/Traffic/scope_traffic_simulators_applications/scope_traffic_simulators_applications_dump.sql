BEGIN TRANSACTION;
CREATE TABLE Applications (
    application_id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_name TEXT
);
INSERT INTO "Applications" VALUES(1,'Urban_Simulation');
INSERT INTO "Applications" VALUES(2,'Highway_Simulation');
INSERT INTO "Applications" VALUES(3,'Rural_Simulation');
INSERT INTO "Applications" VALUES(4,'Parking_Simulation');
INSERT INTO "Applications" VALUES(5,'Weather_Impact_Simulation');
CREATE TABLE Traffic_Simulators (
    simulator_id INTEGER PRIMARY KEY AUTOINCREMENT,
    simulator_name TEXT,
    simulator_version TEXT,
    FOREIGN KEY (simulator_id) REFERENCES Traffic_Simulators_Applications(traffic_simulator_id)
);
INSERT INTO "Traffic_Simulators" VALUES(1,'Sim1','V1.0');
INSERT INTO "Traffic_Simulators" VALUES(2,'Sim2','V2.0');
INSERT INTO "Traffic_Simulators" VALUES(3,'Sim3','V3.0');
INSERT INTO "Traffic_Simulators" VALUES(4,'Sim4','V4.0');
INSERT INTO "Traffic_Simulators" VALUES(5,'Sim5','V5.0');
CREATE TABLE Traffic_Simulators_Applications (
    traffic_simulator_id INTEGER,
    application_id INTEGER,
    FOREIGN KEY (traffic_simulator_id) REFERENCES Traffic_Simulators(simulator_id),
    FOREIGN KEY (application_id) REFERENCES Applications(application_id)
);
INSERT INTO "Traffic_Simulators_Applications" VALUES(1,1);
INSERT INTO "Traffic_Simulators_Applications" VALUES(2,1);
INSERT INTO "Traffic_Simulators_Applications" VALUES(3,1);
INSERT INTO "Traffic_Simulators_Applications" VALUES(4,1);
INSERT INTO "Traffic_Simulators_Applications" VALUES(5,1);
INSERT INTO "Traffic_Simulators_Applications" VALUES(5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Applications',5);
INSERT INTO "sqlite_sequence" VALUES('Traffic_Simulators',5);
COMMIT;
