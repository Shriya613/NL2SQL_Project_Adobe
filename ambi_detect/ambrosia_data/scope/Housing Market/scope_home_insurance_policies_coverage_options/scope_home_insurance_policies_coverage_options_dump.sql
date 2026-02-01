BEGIN TRANSACTION;
CREATE TABLE Coverage_Options(
    OptionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT
);
INSERT INTO "Coverage_Options" VALUES(1,'Fire Damage');
INSERT INTO "Coverage_Options" VALUES(2,'Theft');
INSERT INTO "Coverage_Options" VALUES(3,'Flood');
INSERT INTO "Coverage_Options" VALUES(4,'Earthquake');
INSERT INTO "Coverage_Options" VALUES(5,'Windstorm');
CREATE TABLE Home_Insurance_Policies(
    PolicyID INTEGER PRIMARY KEY AUTOINCREMENT,
    HomeAddress TEXT,
    InsuredAmount REAL
);
INSERT INTO "Home_Insurance_Policies" VALUES(1,'123 Main St',300000.0);
INSERT INTO "Home_Insurance_Policies" VALUES(2,'456 Elm St',400000.0);
INSERT INTO "Home_Insurance_Policies" VALUES(3,'789 Oak St',500000.0);
INSERT INTO "Home_Insurance_Policies" VALUES(4,'999 Pine St',600000.0);
INSERT INTO "Home_Insurance_Policies" VALUES(5,'101 Maple Ave',700000.0);
CREATE TABLE Home_Insurance_Policies_Coverage_Options(
    PolicyID INTEGER,
    OptionID INTEGER,
    FOREIGN KEY (PolicyID) REFERENCES Home_Insurance_Policies(PolicyID),
    FOREIGN KEY (OptionID) REFERENCES Coverage_Options(OptionID),
    PRIMARY KEY (PolicyID, OptionID)
);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(1,1);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(1,2);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(2,1);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(2,3);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(3,1);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(3,4);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(4,2);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(4,5);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(5,1);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(5,3);
INSERT INTO "Home_Insurance_Policies_Coverage_Options" VALUES(4,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Coverage_Options',5);
INSERT INTO "sqlite_sequence" VALUES('Home_Insurance_Policies',5);
COMMIT;
