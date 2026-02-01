BEGIN TRANSACTION;
CREATE TABLE Land_Trusts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trustName TEXT,
    location TEXT);
INSERT INTO "Land_Trusts" VALUES(1,'Forest Trust','New York');
INSERT INTO "Land_Trusts" VALUES(2,'Water Trust','California');
INSERT INTO "Land_Trusts" VALUES(3,'Urban Trust','Texas');
INSERT INTO "Land_Trusts" VALUES(4,'Rural Trust','Florida');
INSERT INTO "Land_Trusts" VALUES(5,'Coastal Trust','Washington');
CREATE TABLE Land_Trusts_Properties(
    landTrustId INTEGER,
    propertyId INTEGER,
    FOREIGN KEY (landTrustId) REFERENCES Land_Trusts(id),
    FOREIGN KEY (propertyId) REFERENCES Properties(propertyId),
    PRIMARY KEY (landTrustId, propertyId));
INSERT INTO "Land_Trusts_Properties" VALUES(1,1);
INSERT INTO "Land_Trusts_Properties" VALUES(2,2);
INSERT INTO "Land_Trusts_Properties" VALUES(3,3);
INSERT INTO "Land_Trusts_Properties" VALUES(4,4);
INSERT INTO "Land_Trusts_Properties" VALUES(5,5);
INSERT INTO "Land_Trusts_Properties" VALUES(2,1);
INSERT INTO "Land_Trusts_Properties" VALUES(3,1);
INSERT INTO "Land_Trusts_Properties" VALUES(4,1);
INSERT INTO "Land_Trusts_Properties" VALUES(5,1);
CREATE TABLE "Properties"(
    propertyId INTEGER PRIMARY KEY AUTOINCREMENT,
    typeOfProperty TEXT
);
INSERT INTO "Properties" VALUES(1,'Conservation Easement');
INSERT INTO "Properties" VALUES(2,'Residential Property');
INSERT INTO "Properties" VALUES(3,'Commercial Property');
INSERT INTO "Properties" VALUES(4,'Agriculture Property');
INSERT INTO "Properties" VALUES(5,'Industrial Property');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Land_Trusts',5);
INSERT INTO "sqlite_sequence" VALUES('Properties',5);
COMMIT;
