BEGIN TRANSACTION;
CREATE TABLE "RacquetsAndTournaments"(
    ID INTEGER PRIMARY KEY AUTOINCREMENT, EventName TEXT,
    EventType TEXT ,
    Sponsor TEXT,
    Participants INT,
    PrizeMoney FLOAT
);
INSERT INTO "RacquetsAndTournaments" VALUES(1,'Summer Grand Slam','Tennis Tournament','Nike',64,1000000.0);
INSERT INTO "RacquetsAndTournaments" VALUES(2,'Reebok Open','Tennis Tournament','Reebok',64,1000000.0);
INSERT INTO "RacquetsAndTournaments" VALUES(3,'Nike Golf Classic','Golf Tournament','Nike',128,2000000.0);
INSERT INTO "RacquetsAndTournaments" VALUES(4,'Under Armour Invitational','Golf Tournament','Under Armour',128,2000000.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('RacquetsAndTournaments',4);
COMMIT;
