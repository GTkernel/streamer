#ifndef framesdatabase_hpp
#define framesdatabase_hpp
#include "litesql.hpp"
#include <memory>
class FrameEntry;
class FrameEntry : public litesql::Persistent {
public:
    class Own {
    public:
        static const litesql::FieldType Id;
    };
    static const std::string type__;
    static const std::string table__;
    static const std::string sequence__;
    static const litesql::FieldType Id;
    litesql::Field<int> id;
    static const litesql::FieldType Type;
    litesql::Field<std::string> type;
    static const litesql::FieldType Path;
    litesql::Field<std::string> path;
    static const litesql::FieldType Date;
    litesql::Field<litesql::DateTime> date;
    static const litesql::FieldType Exposure;
    litesql::Field<float> exposure;
    static const litesql::FieldType Sharpness;
    litesql::Field<float> sharpness;
    static const litesql::FieldType Brightness;
    litesql::Field<float> brightness;
    static const litesql::FieldType Saturation;
    litesql::Field<float> saturation;
    static const litesql::FieldType Hue;
    litesql::Field<float> hue;
    static const litesql::FieldType Gain;
    litesql::Field<float> gain;
    static const litesql::FieldType Gamma;
    litesql::Field<float> gamma;
    static const litesql::FieldType Wbred;
    litesql::Field<float> wbred;
    static const litesql::FieldType Wbblue;
    litesql::Field<float> wbblue;
    static void initValues();
protected:
    void defaults();
public:
    FrameEntry(const litesql::Database& db);
    FrameEntry(const litesql::Database& db, const litesql::Record& rec);
    FrameEntry(const FrameEntry& obj);
    const FrameEntry& operator=(const FrameEntry& obj);
protected:
    std::string insert(litesql::Record& tables, litesql::Records& fieldRecs, litesql::Records& valueRecs);
    void create();
    virtual void addUpdates(Updates& updates);
    virtual void addIDUpdates(Updates& );
public:
    static void getFieldTypes(std::vector<litesql::FieldType>& ftypes);
protected:
    virtual void delRecord();
    virtual void delRelations();
public:
    virtual void update();
    virtual void del();
    virtual bool typeIsCorrect() const;
    std::unique_ptr<FrameEntry> upcast() const;
    std::unique_ptr<FrameEntry> upcastCopy() const;
};
std::ostream & operator<<(std::ostream& os, FrameEntry const& o);
class FramesDatabase : public litesql::Database {
public:
    FramesDatabase(std::string backendType, std::string connInfo);
protected:
    virtual std::vector<litesql::Database::SchemaItem> getSchema() const;
    static void initialize();
};
#endif
