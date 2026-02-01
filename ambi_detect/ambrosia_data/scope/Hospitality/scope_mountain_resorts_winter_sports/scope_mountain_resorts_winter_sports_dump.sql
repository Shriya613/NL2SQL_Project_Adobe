BEGIN TRANSACTION;
CREATE TABLE Mountain_Resorts(
    ResortID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    NumOfRooms INTEGER
);
INSERT INTO "Mountain_Resorts" VALUES(1,'Alpine Lodge','Colorado Mountains',100);
INSERT INTO "Mountain_Resorts" VALUES(2,'Aspen Ski Chalet','Colorado Mountains',200);
INSERT INTO "Mountain_Resorts" VALUES(3,'Rocky Peak Inn','Utah Mountains',300);
INSERT INTO "Mountain_Resorts" VALUES(4,'High Country Retreat','Montana Mountains',400);
INSERT INTO "Mountain_Resorts" VALUES(5,'Peak View Motel','California Mountains',500);
CREATE TABLE Mountain_Resorts_Winter_Sports(
    ResortID INTEGER REFERENCES Mountain_Resorts(ResortID),
    SportID INTEGER REFERENCES Winter_Sports(SportID)
);
INSERT INTO "Mountain_Resorts_Winter_Sports" VALUES(1,1);
INSERT INTO "Mountain_Resorts_Winter_Sports" VALUES(2,2);
INSERT INTO "Mountain_Resorts_Winter_Sports" VALUES(3,3);
INSERT INTO "Mountain_Resorts_Winter_Sports" VALUES(4,4);
INSERT INTO "Mountain_Resorts_Winter_Sports" VALUES(5,5);
INSERT INTO "Mountain_Resorts_Winter_Sports" VALUES(2,1);
INSERT INTO "Mountain_Resorts_Winter_Sports" VALUES(3,1);
INSERT INTO "Mountain_Resorts_Winter_Sports" VALUES(4,1);
INSERT INTO "Mountain_Resorts_Winter_Sports" VALUES(5,1);
CREATE TABLE Winter_Sports(
    SportID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT
);
INSERT INTO "Winter_Sports" VALUES(1,'Snowboarding','A thrilling snow sport that involves descending a snow-covered slope on a snowboard');
INSERT INTO "Winter_Sports" VALUES(2,'Cross-Country Skiing','A form of skiing where participants travel over long distances across snow-covered terrain');
INSERT INTO "Winter_Sports" VALUES(3,'Downhill Skiing','A popular alpine winter sport involving sliding down snow-covered slopes on skis with fixed bindings to boots');
INSERT INTO "Winter_Sports" VALUES(4,'Ice Skating','A recreational activity where people wear ice skates to glide on an ice surface');
INSERT INTO "Winter_Sports" VALUES(5,'Ice Hockey','An athletic team sport played on ice, in which skaters use sticks to shoot a flat puck into the opposing goal');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Mountain_Resorts',5);
INSERT INTO "sqlite_sequence" VALUES('Winter_Sports',5);
COMMIT;
