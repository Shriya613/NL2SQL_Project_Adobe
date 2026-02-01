BEGIN TRANSACTION;
CREATE TABLE Homeowners_association_rules(
    associd INTEGER,
    ruleid INTEGER,
    FOREIGN KEY (associd) REFERENCES Homeowners_associations(id),
    FOREIGN KEY (ruleid) REFERENCES Rules(ruleid),
    PRIMARY KEY (associd, ruleid)
);
INSERT INTO "Homeowners_association_rules" VALUES(1,1);
INSERT INTO "Homeowners_association_rules" VALUES(2,1);
INSERT INTO "Homeowners_association_rules" VALUES(3,1);
INSERT INTO "Homeowners_association_rules" VALUES(4,1);
INSERT INTO "Homeowners_association_rules" VALUES(5,1);
INSERT INTO "Homeowners_association_rules" VALUES(5,2);
CREATE TABLE Homeowners_associations(
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT
);
INSERT INTO "Homeowners_associations" VALUES(1,'Sunnyvale HOA','California');
INSERT INTO "Homeowners_associations" VALUES(2,'Rainbow Hills Association','Florida');
INSERT INTO "Homeowners_associations" VALUES(3,'Cherry Blossom Club','Texas');
INSERT INTO "Homeowners_associations" VALUES(4,'Evergreen Estates','New York');
INSERT INTO "Homeowners_associations" VALUES(5,'Diamond Cove Community','Washington');
CREATE TABLE Homeowners_associations_homes(
    associd INTEGER,
    houseid INTEGER,
    FOREIGN KEY (associd) REFERENCES Homeowners_associations(id),
    FOREIGN KEY (houseid) REFERENCES Homes(houseid),
    PRIMARY KEY (associd, houseid)
);
INSERT INTO "Homeowners_associations_homes" VALUES(1,1);
INSERT INTO "Homeowners_associations_homes" VALUES(2,2);
INSERT INTO "Homeowners_associations_homes" VALUES(3,3);
INSERT INTO "Homeowners_associations_homes" VALUES(4,4);
INSERT INTO "Homeowners_associations_homes" VALUES(5,5);
CREATE TABLE Homes(
    houseid INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT,
    price REAL
);
INSERT INTO "Homes" VALUES(1,'123 Main St',300000.0);
INSERT INTO "Homes" VALUES(2,'456 Oak Ave',400000.0);
INSERT INTO "Homes" VALUES(3,'789 Pine Rd',500000.0);
INSERT INTO "Homes" VALUES(4,'101 Elm Ln',600000.0);
INSERT INTO "Homes" VALUES(5,'202 Maple Dr',700000.0);
CREATE TABLE Rules(
    ruleid INTEGER PRIMARY KEY AUTOINCREMENT,
    rulename TEXT,
    description TEXT
);
INSERT INTO "Rules" VALUES(1,'Noise Ordinance','Respect quiet hours from 10 PM - 7 AM.');
INSERT INTO "Rules" VALUES(2,'Parking Regulations','All vehicles must park on driveways only.');
INSERT INTO "Rules" VALUES(3,'Landscaping Guidelines','Maintain a well-manicured lawn and trimmed hedges.');
INSERT INTO "Rules" VALUES(4,'Pet Restrictions','Only small dogs allowed, no aggressive breeds.');
INSERT INTO "Rules" VALUES(5,'Architectural Review Process','Submit plans for any exterior modifications.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Rules',5);
INSERT INTO "sqlite_sequence" VALUES('Homes',5);
COMMIT;
