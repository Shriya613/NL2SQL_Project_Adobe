BEGIN TRANSACTION;
CREATE TABLE cocktail_lounges(
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT,
    phone_number TEXT,
    hours VARCHAR(100)
);
INSERT INTO "cocktail_lounges" VALUES(1,'The Mixologist','123 Main St','555-1234','Open from 5 PM to 2 AM');
INSERT INTO "cocktail_lounges" VALUES(2,'The Rooftop Lounge','456 Market St','555-5678','Closed at midnight');
CREATE TABLE employees(
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    position TEXT,
    salary REAL
);
INSERT INTO "employees" VALUES(1,'Jane','Doe','Manager',80000.0);
INSERT INTO "employees" VALUES(2,'John','Smith','Bartender',60000.0);
CREATE TABLE restaurants(
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT,
    phone_number TEXT,
    cuisine TEXT
);
INSERT INTO "restaurants" VALUES(1,'La Casa Italiana','321 Ocean Drive','555-1123','Italian');
INSERT INTO "restaurants" VALUES(2,'Yum Cha Palace','654 Second Street','555-1415','Chinese');
CREATE TABLE rooms(
    id INTEGER PRIMARY KEY,
    hotel_name TEXT,
    room_type TEXT,
    price REAL,
    capacity INTEGER
);
INSERT INTO "rooms" VALUES(1,'Grand Hotel','Deluxe Suite',200.0,2);
INSERT INTO "rooms" VALUES(2,'Coastal Inn','Standard Room',100.0,1);
CREATE TABLE sports_bars(
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT,
    phone_number TEXT,
    hours VARCHAR(100)
);
INSERT INTO "sports_bars" VALUES(1,'Sports Central','789 Park Ave','555-9012','Open from 5 PM to 2 AM');
INSERT INTO "sports_bars" VALUES(2,'Game On!','987 Lake Blvd','555-1357','Opens at noon, closes at 1 AM');
COMMIT;
