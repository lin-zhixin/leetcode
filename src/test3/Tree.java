package test3;


import java.util.*;
import java.util.stream.Collectors;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class Node {
    public int val;
    public Node left;
    public Node right;

    public Node() {
    }

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};

public class Tree {
    //236. 二叉树的最近公共祖先
    TreeNode lowestCommonAncestor = null;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (Objects.equals(root, p) || Objects.equals(root, q)) {
            return root;
        }
        TreeNode l = lowestCommonAncestor(root.left, p, q);
        TreeNode r = lowestCommonAncestor(root.right, p, q);
        if (l != null && r != null) {
            return root;
        }
        if (l != null) {
            return l;
        }
        return r;
    }

//    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q)


    //    LCR 155. 将二叉搜索树转化为排序的双向链表（非递归）
    public Node treeToDoublyList(Node root) {
        if (root == null) {
            return null;
        }
        Stack<Node> stack = new Stack<>();
        Node p = root, pre = null, head = null, peek;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }

            peek = stack.pop();
//            做连接
            if (pre == null) {
                head = peek;
            } else {
                pre.right = peek;
            }
            peek.left = pre;
            pre = peek;
//            转换p的值
            p = peek.right;
        }
//        连接头尾
        pre.right = head;
        head.left = pre;

        return head;
    }

    //    LCR 155. 将二叉搜索树转化为排序的双向链表（递归）
    Node pre, head;

    public Node treeToDoublyList2(Node root) {
        if (root == null) {
            return null;
        }
        treeToDoublyList21(root);
        head.left = pre;
        pre.right = head;
        return head;
    }

    public void treeToDoublyList21(Node root) {
        if (root == null) {
            return;
        }
        treeToDoublyList21(root.left);
        if (pre == null) {
            head = root;
        } else {
            pre.right = root;
        }
        root.left = pre;
        pre = root;
        treeToDoublyList21(root.right);
    }

    //1325. 删除给定值的叶子节点
    public TreeNode removeLeafNodes(TreeNode root, int target) {
        if (root == null) {
            return null;
        }
        if (root.left == null && root.right == null && root.val == target) {
            return root = null;
        }
        root.left = removeLeafNodes(root.left, target);
        root.right = removeLeafNodes(root.right, target);
        if (root.left == null && root.right == null && root.val == target) {
            return root = null;
        }
        return root;
    }

    //    662. 二叉树最大宽度
    public int widthOfBinaryTree(TreeNode root) {
        int max = 0;
        Deque<TreeNode> q = new LinkedList<>();
        Map<TreeNode, Integer> map = new HashMap<>();
        q.offerLast(root);
        map.put(root, 1);
        while (!q.isEmpty()) {
            max = Math.max(max, map.get(q.peekLast()) - map.get(q.peekFirst()) + 1);
            int levelSize = q.size();
//            System.out.println(levelSize);
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = q.pollFirst();
                if (node.left != null) {
                    q.offerLast(node.left);
                    map.put(node.left, map.get(node) * 2);
                }
                if (node.right != null) {
                    q.offerLast(node.right);
                    map.put(node.right, map.get(node) * 2 + 1);
                }
            }

        }
        return max;
    }

    //103. 二叉树的锯齿形层序遍历
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        boolean re = false;
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Deque<TreeNode> q = new LinkedList<>();
        q.offerLast(root);
        while (!q.isEmpty()) {
            List<Integer> t = q.stream().map(e -> e.val).collect(Collectors.toList());
            if (re) {
                Collections.reverse(t);
            }
            res.add(t);
            re = !re;
            int levelSize = q.size();
//            System.out.println(levelSize);
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = q.pollFirst();
                if (node.left != null) {
                    q.offerLast(node.left);
                }
                if (node.right != null) {
                    q.offerLast(node.right);
                }
            }
        }
        return res;
    }
}
