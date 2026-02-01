BEGIN TRANSACTION;
CREATE TABLE Company (
    id INTEGER PRIMARY KEY,
    name TEXT
);
INSERT INTO "Company" VALUES(1,'Company1');
INSERT INTO "Company" VALUES(2,'Company2');
INSERT INTO "Company" VALUES(3,'Company3');
INSERT INTO "Company" VALUES(4,'Company4');
INSERT INTO "Company" VALUES(5,'Company5');
CREATE TABLE MarketingSpecialists (
    id INTEGER PRIMARY KEY,
    position_id INTEGER,
    target_market_id INTEGER,
    skills TEXT,
    experience INT,
    FOREIGN KEY(position_id) REFERENCES Position(id),
    FOREIGN KEY(target_market_id) REFERENCES TargetMarket(id)
);
INSERT INTO "MarketingSpecialists" VALUES(1,4,1,'Skill4',8);
INSERT INTO "MarketingSpecialists" VALUES(2,5,2,'Skill5',11);
CREATE TABLE Position (
    id INTEGER PRIMARY KEY,
    title TEXT,
    salary REAL,
    FOREIGN KEY(id) REFERENCES Company(id)
);
INSERT INTO "Position" VALUES(1,'Position1',60000.0);
INSERT INTO "Position" VALUES(2,'Position2',70000.0);
INSERT INTO "Position" VALUES(3,'Position3',80000.0);
INSERT INTO "Position" VALUES(4,'Position4',90000.0);
INSERT INTO "Position" VALUES(5,'Position5',100000.0);
CREATE TABLE SalesAssociates (
    id INTEGER PRIMARY KEY,
    position_id INTEGER,
    target_market_id INTEGER,
    skills TEXT,
    experience INT,
    FOREIGN KEY(position_id) REFERENCES Position(id),
    FOREIGN KEY(target_market_id) REFERENCES TargetMarket(id)
);
INSERT INTO "SalesAssociates" VALUES(1,1,1,'Skill1',2);
INSERT INTO "SalesAssociates" VALUES(2,2,2,'Skill2',5);
INSERT INTO "SalesAssociates" VALUES(3,3,2,'Skill3',7);
CREATE TABLE TargetMarket (
    id INTEGER PRIMARY KEY,
    type TEXT 
);
INSERT INTO "TargetMarket" VALUES(1,'B2B');
INSERT INTO "TargetMarket" VALUES(2,'B2C');
COMMIT;
