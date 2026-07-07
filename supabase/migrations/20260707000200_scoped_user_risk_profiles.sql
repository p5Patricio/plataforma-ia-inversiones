-- Allow user risk profiles to target default, asset class, or ticker scopes.
alter table user_risk_profiles
  add column if not exists scope_type text not null default 'default';

alter table user_risk_profiles
  add column if not exists scope_value text not null default '';

update user_risk_profiles
set scope_type = 'default',
    scope_value = ''
where is_default = true
  and (scope_type is null or scope_type = 'default');

alter table user_risk_profiles
  drop constraint if exists user_risk_profiles_scope_check;

alter table user_risk_profiles
  add constraint user_risk_profiles_scope_check
  check (
    (scope_type = 'default' and scope_value = '')
    or (scope_type in ('asset_class', 'ticker') and scope_value <> '')
  );

alter table user_risk_profiles
  drop constraint if exists user_risk_profiles_user_id_name_key;

drop index if exists user_risk_profiles_scope_idx;
create unique index user_risk_profiles_scope_idx
  on user_risk_profiles(user_id, scope_type, scope_value);

drop index if exists user_risk_profiles_one_default_idx;
create unique index user_risk_profiles_one_default_idx
  on user_risk_profiles(user_id)
  where is_default and scope_type = 'default';
