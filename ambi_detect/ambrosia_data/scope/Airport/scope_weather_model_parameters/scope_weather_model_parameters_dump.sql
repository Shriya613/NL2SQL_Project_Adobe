BEGIN TRANSACTION;
CREATE TABLE Parameters(
    parameterID INTEGER PRIMARY KEY,
    parameterName TEXT,
    unit TEXT);
INSERT INTO "Parameters" VALUES(1,'Temperature','°C');
INSERT INTO "Parameters" VALUES(2,'Wind Speed','km/h');
INSERT INTO "Parameters" VALUES(3,'Humidity','%');
INSERT INTO "Parameters" VALUES(4,'Visibility','Km');
CREATE TABLE Weather_Model(
    modelID INTEGER PRIMARY KEY,
    modelName TEXT,
    manufacturer TEXT);
INSERT INTO "Weather_Model" VALUES(1,'Model A','Manufacturer X');
INSERT INTO "Weather_Model" VALUES(2,'Model B','Manufacturer Y');
INSERT INTO "Weather_Model" VALUES(3,'Model C','Manufacturer Z');
INSERT INTO "Weather_Model" VALUES(4,'Model D','Manufacturer W');
INSERT INTO "Weather_Model" VALUES(5,'Model E','Manufacturer V');
CREATE TABLE Weather_Model_Parameters(
    weatherModelId INTEGER,
    parameterId INTEGER,
    value REAL,
    FOREIGN KEY (weatherModelId) REFERENCES Weather_Model(modelID),
    FOREIGN KEY (parameterId) REFERENCES Parameters(parameterID));
INSERT INTO "Weather_Model_Parameters" VALUES(1,1,NULL);
INSERT INTO "Weather_Model_Parameters" VALUES(2,1,NULL);
INSERT INTO "Weather_Model_Parameters" VALUES(3,1,NULL);
INSERT INTO "Weather_Model_Parameters" VALUES(4,1,NULL);
INSERT INTO "Weather_Model_Parameters" VALUES(5,1,NULL);
INSERT INTO "Weather_Model_Parameters" VALUES(5,4,NULL);
COMMIT;
