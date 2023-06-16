package category;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class RandomizedSet {

    //    List实现o(1)查询，通过把值放在末尾或交换到末尾来实现插入删除
    List<Integer> list = new ArrayList<>();
    //    记录index
    Map<Integer, Integer> map = new HashMap<>();

    public RandomizedSet() {

    }

    public boolean insert(int val) {
        boolean result = false;
        if (!list.contains(val)) {
            list.add(val);
            map.put(val, list.size() - 1);
            result = true;
        }
        System.out.println(list);
        return result;
    }

    public boolean remove(int val) {
        if (!map.containsKey(val)) {
            return false;
        }
//        if (map.get(val)==list.size()-1){
//
//        }
        Integer ind1 = map.get(val), ind2 = list.size() - 1;
        map.put(list.get(ind2), ind1);
        Integer t = list.get(ind1);
        list.set(ind1, list.get(ind2));
        list.set(ind2, t);
        list.remove(ind2);
        map.remove(val);
        System.out.println(list);

        return true;
    }

    public int getRandom() {
        return list.get((int) (Math.random() * list.size()));

    }
}

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet obj = new RandomizedSet();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */