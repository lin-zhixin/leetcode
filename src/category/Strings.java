package category;

import java.util.*;

// 字符串处理
public class Strings {

//    3. 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        if (Objects.isNull(s)||s.length()==0)return 0;
        Map<Character,Integer> map=new HashMap<>();
        Integer left=0;
        Integer max=0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))){
//                left=map.get(s.charAt(i))+1;
                left=Math.max(left,map.get(s.charAt(i))+1);//left要取最大值

            }
            map.put(s.charAt(i),i);
            max=Math.max(max,i-left+1);
        }
        return max;
    }

//    49. 字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
//       每个字符串排序确定唯一性 之后再放入map
      Map<String,List<String>> map=new HashMap<>();
        for (String s:strs) {
            char[] chars=s.toCharArray();
            Arrays.sort(chars);
            String key=new String(chars);
            List<String> tempList=map.getOrDefault(key,new ArrayList<String>());
            tempList.add(s);
            map.put(key,tempList);
        }
        return new ArrayList<List<String>>(map.values());
    }



    public static void main(String[] args) {
        Strings obj=new Strings();
        System.out.printf(""+obj.groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}));

    }
}
