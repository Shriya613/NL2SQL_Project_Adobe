BEGIN TRANSACTION;
CREATE TABLE Districts (
    districtID INTEGER PRIMARY KEY,
    name VARCHAR(255),
    totalPopulation INT,
    FOREIGN KEY (districtID) REFERENCES Dominating_Party(districtID),
    FOREIGN KEY (districtID) REFERENCES Prevalent_Ideology(districtID)
);
INSERT INTO "Districts" VALUES(1,'District A',100000);
INSERT INTO "Districts" VALUES(2,'District B',90000);
INSERT INTO "Districts" VALUES(3,'District C',80000);
INSERT INTO "Districts" VALUES(4,'District D',70000);
INSERT INTO "Districts" VALUES(5,'District E',60000);
CREATE TABLE Dominating_Party (
    districtID INTEGER,
    partyName VARCHAR(255),
    percentageVotes DOUBLE,
    PRIMARY KEY (districtID, partyName),
    FOREIGN KEY (districtID) REFERENCES Districts(districtID)
);
INSERT INTO "Dominating_Party" VALUES(1,'Democratic Party',45.6);
INSERT INTO "Dominating_Party" VALUES(2,'Republican Party',49.2);
INSERT INTO "Dominating_Party" VALUES(3,'Libertarian Party',3.1);
INSERT INTO "Dominating_Party" VALUES(4,'Green Party',1.7);
INSERT INTO "Dominating_Party" VALUES(5,'Independents',0.8);
CREATE TABLE Individuals (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INT
);
INSERT INTO "Individuals" VALUES(1,'John Doe',30);
INSERT INTO "Individuals" VALUES(2,'Jane Smith',28);
INSERT INTO "Individuals" VALUES(3,'Alice Johnson',42);
INSERT INTO "Individuals" VALUES(4,'Bob Brown',56);
INSERT INTO "Individuals" VALUES(5,'Carol White',37);
CREATE TABLE Prevalent_Ideology (
    districtID INTEGER,
    ideologyName VARCHAR(255),
    percentageVotes DOUBLE,
    PRIMARY KEY (districtID, ideologyName),
    FOREIGN KEY (districtID) REFERENCES Districts(districtID)
);
INSERT INTO "Prevalent_Ideology" VALUES(1,'Conservative Ideas',25.4);
INSERT INTO "Prevalent_Ideology" VALUES(2,'Progressive Ideas',28.7);
INSERT INTO "Prevalent_Ideology" VALUES(3,'Moderate Ideas',30.6);
INSERT INTO "Prevalent_Ideology" VALUES(4,'Liberal Ideas',15.3);
INSERT INTO "Prevalent_Ideology" VALUES(5,'Radical Ideas',3.8);
CREATE TABLE VotingRecords (
    recordID INTEGER PRIMARY KEY AUTOINCREMENT,
    individualId INTEGER,
    party VARCHAR(255),
    FOREIGN KEY (individualId) REFERENCES Individuals(id)
);
INSERT INTO "VotingRecords" VALUES(1,1,'Democrat');
INSERT INTO "VotingRecords" VALUES(2,2,'Republican');
INSERT INTO "VotingRecords" VALUES(3,3,'Libertarian');
INSERT INTO "VotingRecords" VALUES(4,4,'Green Party');
INSERT INTO "VotingRecords" VALUES(5,5,'Independent');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('VotingRecords',5);
COMMIT;
