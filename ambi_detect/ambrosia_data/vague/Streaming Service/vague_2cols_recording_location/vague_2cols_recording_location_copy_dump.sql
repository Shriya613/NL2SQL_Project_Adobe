BEGIN TRANSACTION;
CREATE TABLE ContentLibrary (
    content_id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    duration INTEGER,
    release_date DATE,
    category TEXT
);
INSERT INTO "ContentLibrary" VALUES(1,'Movie A','A thrilling action movie',120,'2022-01-01','Action');
INSERT INTO "ContentLibrary" VALUES(2,'TV Show B','An exciting crime drama series',480,'2021-10-01','Crime Drama');
CREATE TABLE Customer (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    password TEXT,
    subscription_plan_id INTEGER,
    FOREIGN KEY(subscription_plan_id) REFERENCES SubscriptionPlan(plan_id)
);
INSERT INTO "Customer" VALUES(1,'John','Doe','john@example.com','password123',1);
INSERT INTO "Customer" VALUES(2,'Jane','Smith','jane@example.com','passwordabc',2);
CREATE TABLE PlayHistory (
    history_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    content_id INTEGER,
    play_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(customer_id) REFERENCES Customer(customer_id),
    FOREIGN KEY(content_id) REFERENCES ContentLibrary(content_id)
);
INSERT INTO "PlayHistory" VALUES(1,1,1,'2022-01-02 12:00:00');
INSERT INTO "PlayHistory" VALUES(2,2,2,'2022-01-03 15:30:00');
CREATE TABLE "Studio" (
    id INTEGER PRIMARY KEY,
    Recording_Location TEXT,
    podcast TEXT,
    genre TEXT,
    rating REAL
);
INSERT INTO "Studio" VALUES(1,'Studio A','Podcast XYZ','Comedy',4.8);
INSERT INTO "Studio" VALUES(2,'Studio B','Podcast ABC','Drama',4.7);
CREATE TABLE SubscriptionPlan (
    plan_id INTEGER PRIMARY KEY,
    plan_name TEXT,
    monthly_price REAL,
    annual_price REAL,
    features TEXT
);
INSERT INTO "SubscriptionPlan" VALUES(1,'Basic Plan',9.99,99.0,'Access to all shows');
INSERT INTO "SubscriptionPlan" VALUES(2,'Premium Plan',14.99,149.0,'Ad-free experience');
COMMIT;
