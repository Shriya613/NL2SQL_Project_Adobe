BEGIN TRANSACTION;
CREATE TABLE Airlines(
    AirlineID INTEGER PRIMARY KEY,
    CompanyName TEXT,
    HomeTerminalID INTEGER,
    FOREIGN KEY (HomeTerminalID) REFERENCES Terminals(ID));
INSERT INTO "Airlines" VALUES(1,'Company A',1);
INSERT INTO "Airlines" VALUES(2,'Company B',2);
INSERT INTO "Airlines" VALUES(3,'Company C',3);
INSERT INTO "Airlines" VALUES(4,'Company D',4);
INSERT INTO "Airlines" VALUES(5,'Company E',5);
CREATE TABLE Jet_Bridges(
    ID INTEGER PRIMARY KEY,
    BridgeName TEXT,
    TerminalID INTEGER,
    FOREIGN KEY (TerminalID) REFERENCES Terminals(ID));
INSERT INTO "Jet_Bridges" VALUES(1,'Jet Bridge A',1);
INSERT INTO "Jet_Bridges" VALUES(2,'Jet Bridge B',2);
INSERT INTO "Jet_Bridges" VALUES(3,'Jet Bridge C',3);
INSERT INTO "Jet_Bridges" VALUES(4,'Jet Bridge D',4);
INSERT INTO "Jet_Bridges" VALUES(5,'Jet Bridge E',5);
CREATE TABLE Staircars(
    ID INTEGER PRIMARY KEY,
    CarName TEXT,
    TerminalID INTEGER,
    FOREIGN KEY (TerminalID) REFERENCES Terminals(ID));
INSERT INTO "Staircars" VALUES(1,'Staircar X',1);
INSERT INTO "Staircars" VALUES(2,'Staircar Y',2);
INSERT INTO "Staircars" VALUES(3,'Staircar Z',3);
INSERT INTO "Staircars" VALUES(4,'Staircar W',4);
INSERT INTO "Staircars" VALUES(5,'Staircar V',5);
CREATE TABLE Terminals(
    ID INTEGER PRIMARY KEY,
    Name TEXT UNIQUE);
INSERT INTO "Terminals" VALUES(1,'Terminal 1');
INSERT INTO "Terminals" VALUES(2,'Terminal 2');
INSERT INTO "Terminals" VALUES(3,'Terminal 3');
INSERT INTO "Terminals" VALUES(4,'Terminal 4');
INSERT INTO "Terminals" VALUES(5,'Terminal 5');
COMMIT;
