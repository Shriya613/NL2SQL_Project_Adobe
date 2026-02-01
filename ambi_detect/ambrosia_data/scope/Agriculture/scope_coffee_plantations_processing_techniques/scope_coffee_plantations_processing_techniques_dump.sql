BEGIN TRANSACTION;
CREATE TABLE Coffee_Plantations (
    plantation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Coffee_Plantations" VALUES(1,'Plantation A','Location A');
INSERT INTO "Coffee_Plantations" VALUES(2,'Plantation B','Location B');
INSERT INTO "Coffee_Plantations" VALUES(3,'Plantation C','Location C');
INSERT INTO "Coffee_Plantations" VALUES(4,'Plantation D','Location D');
INSERT INTO "Coffee_Plantations" VALUES(5,'Plantation E','Location E');
CREATE TABLE Coffee_Plantations_Processing_Techniques (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coffee_plantation_id INTEGER,
    processing_technique_id INTEGER,
    FOREIGN KEY(coffee_plantation_id) REFERENCES Coffee_Plantations(plantation_id),
    FOREIGN KEY(processing_technique_id) REFERENCES Processing_Techniques(technique_id));
INSERT INTO "Coffee_Plantations_Processing_Techniques" VALUES(1,1,1);
INSERT INTO "Coffee_Plantations_Processing_Techniques" VALUES(2,1,2);
INSERT INTO "Coffee_Plantations_Processing_Techniques" VALUES(3,1,3);
INSERT INTO "Coffee_Plantations_Processing_Techniques" VALUES(4,1,4);
INSERT INTO "Coffee_Plantations_Processing_Techniques" VALUES(5,1,5);
INSERT INTO "Coffee_Plantations_Processing_Techniques" VALUES(11,2,1);
INSERT INTO "Coffee_Plantations_Processing_Techniques" VALUES(12,3,1);
INSERT INTO "Coffee_Plantations_Processing_Techniques" VALUES(13,4,1);
INSERT INTO "Coffee_Plantations_Processing_Techniques" VALUES(14,5,1);
CREATE TABLE Processing_Techniques (
    technique_id INTEGER PRIMARY KEY AUTOINCREMENT,
    technique_name TEXT);
INSERT INTO "Processing_Techniques" VALUES(1,'Wet_Processing');
INSERT INTO "Processing_Techniques" VALUES(2,'Dry_Processing');
INSERT INTO "Processing_Techniques" VALUES(3,'Honey_Processing');
INSERT INTO "Processing_Techniques" VALUES(4,'Natural_Processing');
INSERT INTO "Processing_Techniques" VALUES(5,'Pulped_Natural_Processing');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Coffee_Plantations',5);
INSERT INTO "sqlite_sequence" VALUES('Processing_Techniques',5);
INSERT INTO "sqlite_sequence" VALUES('Coffee_Plantations_Processing_Techniques',14);
COMMIT;
