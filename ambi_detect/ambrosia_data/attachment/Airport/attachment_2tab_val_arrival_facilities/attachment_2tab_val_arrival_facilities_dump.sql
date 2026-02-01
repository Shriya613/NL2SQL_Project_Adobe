BEGIN TRANSACTION;
CREATE TABLE arrival_hall (
    id INTEGER PRIMARY KEY,
    terminal_id INTEGER REFERENCES terminal(id),
    Arrival_Hall TEXT,
    Maximum_Capacity INT,
    number_of_gates INT
);
INSERT INTO "arrival_hall" VALUES(1,1,'A Hall 1',500,10);
INSERT INTO "arrival_hall" VALUES(2,1,'A Hall 2',600,8);
INSERT INTO "arrival_hall" VALUES(3,2,'B Hall 1',700,12);
INSERT INTO "arrival_hall" VALUES(4,2,'B Hall 2',900,15);
CREATE TABLE baggage_reclaim (
    id INTEGER PRIMARY KEY,
    terminal_id INTEGER REFERENCES terminal(id),
    Baggage_Reclaim TEXT,
    Maximum_Capacity INT,
    number_of_carousels INT
);
INSERT INTO "baggage_reclaim" VALUES(1,1,'A Reclaim 1',500,3);
INSERT INTO "baggage_reclaim" VALUES(2,1,'A Reclaim 2',600,5);
INSERT INTO "baggage_reclaim" VALUES(3,2,'B Reclaim 1',700,4);
INSERT INTO "baggage_reclaim" VALUES(4,2,'B Reclaim 2',900,7);
CREATE TABLE flight (
    id INTEGER PRIMARY KEY,
    terminal_id INTEGER REFERENCES terminal(id)
);
INSERT INTO "flight" VALUES(1,1);
INSERT INTO "flight" VALUES(2,2);
CREATE TABLE terminal (
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT
);
INSERT INTO "terminal" VALUES(1,'Terminal A','Gate A1');
INSERT INTO "terminal" VALUES(2,'Terminal B','Gate B2');
COMMIT;
